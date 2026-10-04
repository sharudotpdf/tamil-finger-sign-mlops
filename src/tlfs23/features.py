"""Handgeometrie, milde Trainingsaugmentation und isolierte MediaPipe-Extraktion."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

from .common import (class_root, digest, paths, read_csv, read_json, require_columns,
                     runtime, safe_path, sha256_file, write_json)

MEDIAPIPE_VERSION = '0.10.35'
EXPECTED_MODEL_SHA256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
TIPS, BASES = [4,8,12,16,20], [1,5,9,13,17]
TRIPLES = [(b,b+1,b+2) for b in BASES] + [(b+1,b+2,b+3) for b in BASES]
HAND_FEATURES = ([f'lm{i:02d}_{a}' for i in range(21) for a in 'xyz']
                 + [f'joint_cos_{i:02d}' for i in range(10)]
                 + [f'tip_wrist_{i}' for i in range(5)]
                 + [f'tip_base_{i}' for i in range(5)]
                 + [f'tip_gap_{i}' for i in range(4)])
FEATURE_NAMES = ([f'h{h}_{name}' for h in range(2) for name in HAND_FEATURES]
                 + ['wrist_dx','wrist_dy','wrist_distance','log_hand_scale_ratio'])


def pair_features(hands: list[np.ndarray], width: int, height: int) -> np.ndarray:
    """h0/h1 folgen der Bildposition, nicht einer angenommenen Vokal-/Konsonantenrolle."""
    if len(hands) != 2:
        raise ValueError('Zwei erkannte Haende erforderlich; fehlende Hand ist keine Faust.')
    vectors, wrists, scales = [], [], []
    for hand in sorted(hands, key=lambda h:(float(h[0,0]),float(h[0,1]))):
        hand = np.asarray(hand, dtype=np.float64)
        if hand.shape != (21,3) or not np.isfinite(hand).all():
            raise ValueError('Ungueltige Handlandmarks.')
        xyz = hand * [width, height, width]
        wrist = xyz[0].copy()
        scale = float(np.linalg.norm(xyz[[5,9,13,17]] - wrist, axis=1).mean())
        if scale < 1e-6:
            raise ValueError('Entartete Handgeometrie.')
        local = (xyz - wrist) / scale
        angles = []
        for a,b,c in TRIPLES:
            u,v = local[a]-local[b], local[c]-local[b]
            denominator = float(np.linalg.norm(u)*np.linalg.norm(v))
            if denominator < 1e-8:
                raise ValueError('Entartetes Fingergelenk.')
            angles.append(float(np.clip(np.dot(u,v)/denominator,-1,1)))
        vector = [*local.ravel(), *angles,
                  *np.linalg.norm(local[TIPS],axis=1),
                  *np.linalg.norm(local[TIPS]-local[BASES],axis=1),
                  *np.linalg.norm(np.diff(local[TIPS],axis=0),axis=1)]
        vectors.extend(vector); wrists.append(wrist); scales.append(scale)
    displacement = (wrists[1]-wrists[0])[:2] / np.mean(scales)
    result = np.asarray([*vectors, *displacement, np.linalg.norm(displacement),
                         np.log(scales[1]/scales[0])],dtype=np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError('Fehlerhafte Feature-Geometrie.')
    return result


def augment(image: Image.Image, sha: str, seed: int) -> tuple[Image.Image, dict]:
    """Reproduzierbar pro Original; keine Spiegelung und kein Abschneiden der Haende."""
    local_seed = int(digest({'image':sha,'seed':seed,'method':'mild_1'})[:16],16)
    rng = np.random.default_rng(local_seed)
    params = {'angle':float(rng.uniform(-6,6)), 'brightness':float(rng.uniform(.9,1.1)),
              'contrast':float(rng.uniform(.9,1.1))}
    transformed = image.rotate(params['angle'],resample=Image.Resampling.BILINEAR,expand=True)
    transformed = ImageEnhance.Brightness(transformed).enhance(params['brightness'])
    transformed = ImageEnhance.Contrast(transformed).enhance(params['contrast'])
    return transformed, params


def samples(frame: pd.DataFrame, augment_train: bool, seed: int = 42) -> pd.DataFrame:
    require_columns(frame,['relative_path','sha256','split','category'])
    if frame.sha256.duplicated().any():
        raise ValueError('Doppelte Originale im Split.')
    result = frame.assign(variant='original').copy()
    if augment_train:
        # Schwerpunkt auf aktiver Einzelkomponente + Faust; Originale bleiben enthalten.
        focus = frame.loc[frame.split.eq('train') & frame.category.isin(['uyir','mei'])]
        result = pd.concat([result,focus.assign(variant='mild_1')],ignore_index=True)
    result['augmentation_seed'] = seed
    result['sample_id'] = [digest({'sha':h,'variant':v,'seed':seed if v!='original' else None})
                           for h,v in zip(result.sha256,result.variant)]
    if result.sample_id.duplicated().any():
        raise ValueError('Doppelte sample_id.')
    return result.sort_values('sample_id').reset_index(drop=True)


class Detector:
    def __init__(self, model: Path, config: dict):
        from importlib.metadata import version
        if version('mediapipe') != MEDIAPIPE_VERSION:
            raise RuntimeError(f'Erwartet mediapipe=={MEDIAPIPE_VERSION}. Notebook 00 pruefen.')
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        self.mp = mp
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model),delegate=BaseOptions.Delegate.CPU),
            running_mode=RunningMode.IMAGE,num_hands=2,
            min_hand_detection_confidence=float(config['detection_threshold']),
            min_hand_presence_confidence=float(config['presence_threshold']))
        self.detector = HandLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray) -> list[np.ndarray]:
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
        result = self.detector.detect(image)
        return [np.asarray([[p.x,p.y,p.z] for p in h],dtype=np.float32) for h in result.hand_landmarks]

    def close(self):
        self.detector.close()


def process_batch(base: Path, frame: pd.DataFrame, detector, config: dict) -> tuple[list,np.ndarray,np.ndarray]:
    records, feature_list, landmark_list = [], [], []
    for number,row in enumerate(frame.to_dict('records'),1):
        start = time.perf_counter()
        record = {'sample_id':row['sample_id'], 'n_hands':-1, 'first_n_hands':-1,
                  'status':'processing_error', 'retry_used':False, 'error':'',
                  'augmentation':'{}', 'image_width':0, 'image_height':0}
        feature = np.full(len(FEATURE_NAMES),np.nan,dtype=np.float32)
        landmark = np.full((2,21,3),np.nan,dtype=np.float32)
        # Geaenderte Rohdaten duerfen nicht als normale Detektionsfehler verschwinden.
        path = safe_path(base,row['relative_path'])
        if sha256_file(path) != row['sha256']:
            raise ValueError(f'Rohdaten geaendert: {row["relative_path"]}')
        try:
            with Image.open(path) as source:
                source.load()
                image = ImageOps.exif_transpose(source).convert('RGB')
            if row['variant'] != 'original':
                if row['split'] != 'train' or row['variant'] != 'mild_1':
                    raise ValueError('Augmentation ausserhalb von Train oder unbekannte Variante.')
                image, params = augment(image,row['sha256'],int(row['augmentation_seed']))
                record['augmentation'] = json.dumps(params,sort_keys=True)
            image.thumbnail((config['max_side'],config['max_side']),Image.Resampling.LANCZOS)
            record['image_width'],record['image_height'] = image.size
            rgb = np.asarray(image,dtype=np.uint8)
            hands = detector.detect(rgb)
            record['first_n_hands'] = len(hands)
            if len(hands)<2 and config['retry']:
                record['retry_used'] = True
                enhanced = ImageEnhance.Contrast(image).enhance(1.1)
                enhanced = ImageEnhance.Brightness(enhanced).enhance(1.1)
                retried = detector.detect(np.asarray(enhanced,dtype=np.uint8))
                # Ein deterministischer Versuch, gleicher Ablauf fuer jede Eingabe.
                if len(retried)>len(hands):
                    hands = retried
            hands = sorted(hands,key=lambda h:(float(h[0,0]),float(h[0,1])))
            record['n_hands'] = len(hands)
            for i,hand in enumerate(hands[:2]):
                landmark[i] = hand
            if len(hands)==2:
                feature = pair_features(hands,image.width,image.height)
                record['status'] = 'ok'
            else:
                record['status'] = 'not_detected' if not hands else 'incomplete_pair'
        except (MemoryError,KeyboardInterrupt):
            raise
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {str(exc)[:250]}'
        record['feature_valid'] = record['status'] == 'ok'
        record['extract_ms'] = float((time.perf_counter()-start)*1000)
        records.append(record); feature_list.append(feature); landmark_list.append(landmark)
        if number%16==0 or number==len(frame):
            print(f'Bilder im Block: {number}/{len(frame)}',flush=True)
    return records,np.asarray(feature_list),np.asarray(landmark_list)


def _save_chunk(out: Path, records: list, features: np.ndarray, landmarks: np.ndarray) -> None:
    temp = out.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream,records=np.array(json.dumps(records)),features=features,landmarks=landmarks)
    temp.replace(out)


def _child(job_file: Path) -> None:
    job = read_json(job_file)
    if sha256_file(Path(job['model'])) != job['model_sha256']:
        raise ValueError('Detektormodell veraendert.')
    detector = Detector(Path(job['model']),job['config'])
    try:
        if job.get('preflight'):
            print('HAND LANDMARKER OK',flush=True)
            return
        frame = pd.DataFrame(job['rows'])
        records,x,lm = process_batch(Path(job['base']),frame,detector,job['config'])
        _save_chunk(Path(job['output']),records,x,lm)
    finally:
        detector.close()


def _launch(root: Path, job_file: Path, log: Path, timeout: int) -> None:
    env = dict(os.environ)
    env['PYTHONPATH'] = str(root / 'src') + os.pathsep + env.get('PYTHONPATH','')
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    command = [sys.executable,'-u','-m','tlfs23.features','--worker',str(job_file)]
    try:
        with log.open('w') as stream:
            result = subprocess.run(command,cwd=root,env=env,stdout=stream,
                                    stderr=subprocess.STDOUT,timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'Zeitlimit fuer diesen Block erreicht. Log: {log}. '
                           'Gespeicherte Bloecke bleiben erhalten; Notebook-Kernel lebt.') from exc
    if result.returncode:
        tail = log.read_text(errors='replace')[-1800:]
        raise RuntimeError(f'MediaPipe-Prozess beendet (Code {result.returncode}). Log: {log}\n{tail}')


def model_path(root: Path) -> Path:
    model = root / 'models/detector/hand_landmarker_v1.task'
    if not model.is_file():
        raise FileNotFoundError(f'Detektordatei fehlt: {model}. Vorhandene Datei uebernehmen; kein automatischer Download.')
    if sha256_file(model) != EXPECTED_MODEL_SHA256:
        raise ValueError('Detektor-Hash weicht vom festgelegten Modell ab. Quelle pruefen.')
    return model


def preflight(root: Path, config: dict, timeout: int = 90) -> None:
    p,model = paths(root),model_path(root)
    job = p['work'] / 'detector_preflight.json'
    log = p['work'] / 'detector_preflight.log'
    write_json(job,{'preflight':True,'model':str(model),'model_sha256':sha256_file(model),'config':config})
    _launch(root,job,log,timeout)
    print('MediaPipe-Initialisierung im separaten Prozess erfolgreich.')


def detection_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby(['category','variant'],dropna=False).agg(
        n_images=('sample_id','size'),
        n_two_hands=('status',lambda x:int(x.eq('ok').sum())),
        n_one_hand=('status',lambda x:int(x.eq('incomplete_pair').sum())),
        n_no_hand=('status',lambda x:int(x.eq('not_detected').sum())),
        n_errors=('status',lambda x:int(x.eq('processing_error').sum())),
        median_ms=('extract_ms','median'),p95_ms=('extract_ms',lambda x:float(x.quantile(.95))))
    summary['two_hand_rate'] = summary.n_two_hands/summary.n_images
    return summary.reset_index()


def run_extraction(root: Path, frame: pd.DataFrame, config: dict, name: str,
                   chunk_size: int = 128, timeout: int = 300,
                   allow_holdout: bool = False) -> tuple[pd.DataFrame,Path]:
    """Ein Prozess pro Block. Cache-Schluessel enthalten Code, Modell, Quelle und Varianten."""
    if frame.empty or frame.sample_id.duplicated().any() or chunk_size<1:
        raise ValueError('Leere/mehrdeutige Auswahl oder ungueltige Blockgroesse.')
    if not allow_holdout and frame.split.eq('test').any():
        raise ValueError('Notebook 05 verarbeitet keine Holdout-Bilder.')
    if ((frame.variant!='original') & (frame.split!='train')).any():
        raise ValueError('Nur Training darf augmentiert werden.')
    p,model = paths(root),model_path(root)
    versions = runtime()
    identity = {'config':config,'model_sha256':sha256_file(model), 'runtime':versions,
                'features_code_sha256':sha256_file(Path(__file__)),
                'feature_names':FEATURE_NAMES,'augmentation':'mild_1 rotation6 brightness10 contrast10',
                'common_code_sha256':sha256_file(Path(__file__).with_name('common.py'))}
    ordered = frame.sort_values('sample_id').reset_index(drop=True)
    selection = digest(ordered.to_dict('records'))
    base = p['work'] / 'feature_runs' / digest(identity)[:16]
    cache = base / 'chunks'; cache.mkdir(parents=True,exist_ok=True)
    output = base / (name + '_' + selection[:12]); output.mkdir(parents=True,exist_ok=True)
    write_json(output / 'request.json',dict(identity,selection_sha256=selection,n_rows=len(ordered)))
    # flock wird nach Prozessende automatisch freigegeben (macOS/Linux).
    import fcntl
    with (base / '.extract.lock').open('w') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Fuer diese Konfiguration laeuft bereits eine Extraktion.') from exc
        files=[]
        for start in range(0,len(ordered),chunk_size):
            part=ordered.iloc[start:start+chunk_size]
            minimal=part[['sample_id','relative_path','sha256','split','variant','augmentation_seed']].to_dict('records')
            token=digest(minimal)
            out=cache / f'{token}.npz'
            if not out.exists():
                job,log=cache / f'{token}.json',cache / f'{token}.log'
                write_json(job,{'model':str(model),'model_sha256':identity['model_sha256'],
                                'config':config,'base':str(class_root(root)),
                                'rows':minimal,'output':str(out)})
                print(f'Extraktion: Bilder {start+1}-{start+len(part)} von {len(ordered)}',flush=True)
                _launch(root,job,log,timeout)
                if not out.exists():
                    raise RuntimeError(f'Block ohne Ergebnis beendet: {log}')
                write_json(out.with_suffix('.sha.json'),{'sha256':sha256_file(out)})
            else:
                print(f'Cache: {min(start+chunk_size,len(ordered))}/{len(ordered)}',flush=True)
                # Inhalt trotz Cache erneut pruefen, Rohdaten muessen unveraendert sein.
                base_images=class_root(root)
                for item in minimal:
                    if sha256_file(safe_path(base_images,item['relative_path']))!=item['sha256']:
                        raise ValueError('Rohdaten seit Audit geaendert.')
            checksum=out.with_suffix('.sha.json')
            if not checksum.exists() or read_json(checksum)['sha256']!=sha256_file(out):
                raise ValueError(f'Unvollstaendiger/veraenderter Cacheblock: {out}. Nur diesen Block archivieren und erneut starten.')
            files.append((out,part.sample_id.tolist()))
        x=np.lib.format.open_memmap(output / 'features.npy',mode='w+',dtype='float32',shape=(len(ordered),len(FEATURE_NAMES)))
        lm=np.lib.format.open_memmap(output / 'landmarks.npy',mode='w+',dtype='float32',shape=(len(ordered),2,21,3))
        records=[]; offset=0
        for file,ids in files:
            with np.load(file,allow_pickle=False) as saved:
                rows=json.loads(saved['records'].item())
                if [r['sample_id'] for r in rows]!=ids or saved['features'].shape!=(len(ids),len(FEATURE_NAMES)):
                    raise ValueError(f'Cache-Reihenfolge ungueltig: {file}')
                x[offset:offset+len(ids)]=saved['features']; lm[offset:offset+len(ids)]=saved['landmarks']
                records.extend(rows); offset+=len(ids)
        x.flush(); lm.flush(); del x,lm
        rows=ordered.merge(pd.DataFrame(records),on='sample_id',validate='1:1',how='left',sort=False)
        rows.to_csv(output / 'rows.csv',index=False)
        detection_summary(rows).to_csv(output / 'detection_summary.csv',index=False)
        write_json(output / 'feature_schema.json',{'names':FEATURE_NAMES,'n_features':len(FEATURE_NAMES),
                  'hand_order':'h0/h1 sorted by image wrist x; no semantic side assignment',
                  'missing_policy':'NaN; status != ok never interpreted as FIST'})
        hashes={f:sha256_file(output/f) for f in ['rows.csv','features.npy','landmarks.npy','feature_schema.json']}
        write_json(output / 'metadata.json',dict(identity,selection_sha256=selection,
                   file_hashes=hashes,n_rows=len(rows),complete=True))
    return rows,output


def show_landmarks(root: Path, rows: pd.DataFrame, output: Path, count: int = 4) -> None:
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown
    landmarks=np.load(output/'landmarks.npy',mmap_mode='r')
    edges=[(0,b) for b in BASES]+[(i,i+1) for b in BASES for i in range(b,b+3)]
    for index in rows.index[:count]:
        row=rows.loc[index]
        display(Markdown(f"**{row['character']} | {row['category']} | {row['variant']}**  \n"
                         f"Status: `{row['status']}`, Haende: {row['n_hands']}, Retry: {row['retry_used']}"))
        with Image.open(safe_path(class_root(root),row['relative_path'])) as source:
            image=ImageOps.exif_transpose(source).convert('RGB')
        if row['variant']!='original':
            image,_=augment(image,row['sha256'],int(row['augmentation_seed']))
        config=read_json(output/'metadata.json')['config']
        image.thumbnail((config['max_side'],config['max_side']),Image.Resampling.LANCZOS)
        fig,ax=plt.subplots(figsize=(5,3.7)); ax.imshow(image)
        for hand in landmarks[index]:
            if np.isfinite(hand).all():
                xx,yy=hand[:,0]*image.width,hand[:,1]*image.height
                ax.scatter(xx,yy,s=12)
                for a,b in edges:
                    ax.plot([xx[a],xx[b]],[yy[a],yy[b]],linewidth=.8)
        ax.axis('off'); fig.tight_layout(); plt.show(); plt.close(fig)


def show_augmentation(root: Path, frame: pd.DataFrame, seed: int = 42, count: int = 2) -> None:
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    if not frame.split.eq('train').all():
        raise ValueError('Augmentierungspruefung nur mit Trainingsbildern.')
    for row in frame.head(count).to_dict('records'):
        with Image.open(safe_path(class_root(root),row['relative_path'])) as source:
            original=ImageOps.exif_transpose(source).convert('RGB')
        changed,params=augment(original,row['sha256'],seed)
        display(Markdown(f"**{row['character']} | {row['category']}** - Parameter: `{params}`"))
        for title,image in [('Original',original),('Leichte Augmentierung',changed)]:
            fig,ax=plt.subplots(figsize=(4.5,3.4))
            ax.imshow(image); ax.set_title(title); ax.axis('off')
            fig.tight_layout(); plt.show(); plt.close(fig)


def save_feature_decision(root: Path, config: dict, approved: bool, note: str, augment_train: bool) -> dict:
    from .data import load_split
    _,state=load_split(root)
    if not approved or len(note.strip())<30 or '...' in note:
        raise ValueError('Detektor und Augmentierung erst pruefen; konkrete Beobachtungen notieren.')
    decision={'config':config,'note':note,'approved':True,'augment_train':augment_train,
              'split_sha256':state['split_sha256'],
              'model_sha256':sha256_file(model_path(root)),
              'feature_names':FEATURE_NAMES,'runtime':runtime(),
              'features_code_sha256':sha256_file(Path(__file__))}
    target=paths(root)['work']/'05_feature_decision.json'
    if target.exists() and read_json(target)!=decision:
        raise ValueError('Feature-Entscheidung bereits festgeschrieben. Alten Stand zuerst archivieren.')
    write_json(target,decision)
    return decision


def save_development_pointer(root: Path, output: Path) -> None:
    p=paths(root)
    decision=read_json(p['work']/'05_feature_decision.json')
    meta=read_json(output/'metadata.json')
    rows=read_csv(output/'rows.csv')
    if not meta['complete'] or rows.split.eq('test').any() or meta['config']!=decision['config']:
        raise ValueError('Entwicklungsartefakt passt nicht zur freigegebenen Pipeline.')
    write_json(p['work']/'development_features.json',
               {'directory':output.relative_to(root).as_posix(),
                'metadata_sha256':sha256_file(output/'metadata.json'),
                'decision_sha256':sha256_file(p['work']/'05_feature_decision.json'),
                'split_sha256':decision['split_sha256']})


def load_development(root: Path):
    from .data import load_split
    _,state=load_split(root)
    p=paths(root)
    pointer=read_json(p['work']/'development_features.json')
    folder=safe_path(root,pointer['directory'])
    if pointer['split_sha256']!=state['split_sha256'] or pointer['decision_sha256']!=sha256_file(p['work']/'05_feature_decision.json'):
        raise ValueError('Feature-Artefakt passt nicht zu Split/Freigabe.')
    if sha256_file(folder/'metadata.json')!=pointer['metadata_sha256']:
        raise ValueError('Feature-Metadaten veraendert.')
    meta=read_json(folder/'metadata.json')
    for name,expected in meta['file_hashes'].items():
        if sha256_file(folder/name)!=expected:
            raise ValueError(f'Artefakt veraendert: {name}')
    rows=read_csv(folder/'rows.csv')
    x=np.load(folder/'features.npy',mmap_mode='r')
    if len(rows)!=len(x) or rows.split.eq('test').any():
        raise ValueError('Ungueltige Entwicklungsdaten.')
    valid=rows.status.eq('ok').to_numpy()
    if not np.isfinite(x[valid]).all() or not np.isnan(x[~valid]).all():
        raise ValueError('Feature-Werte passen nicht zum Detektionsstatus.')
    return rows,x,meta


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',required=True,type=Path)
    args=parser.parse_args()
    _child(args.worker)
