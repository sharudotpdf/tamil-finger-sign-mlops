"""Inventarpruefung, Curation, Labels und festgeschriebene Datenaufteilung."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .common import (as_bool, class_root, digest, folder_key, paths, read_csv,
                     read_json, require_columns, runtime, safe_path,
                     sha256_file, table_digest, write_json)

AUDIT_VERSION = 'rgb-exif-load-1'
BG, NA, FIST = '__background__', '__NA__', 'FIST'


def inventory(root: Path) -> pd.DataFrame:
    frame = read_csv(root / 'data/manifests/file_inventory.csv')
    require_columns(frame, ['relative_path', 'dataset_folder', 'filename', 'file_size_bytes'])
    if frame.empty or frame.relative_path.duplicated().any():
        raise ValueError('Dateiinventar ist leer oder enthaelt doppelte Pfade.')
    return frame.sort_values('relative_path').reset_index(drop=True)


def _image_record(path: Path) -> dict:
    with Image.open(path) as image:
        width, height = image.size
        if width * height > 20_000_000 or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Unerwartete Bildgroesse oder mehrere Frames.')
        fmt, mode = image.format, image.mode
        image.verify()
    # verify() allein dekodiert nicht alle Bildpixel.
    with Image.open(path) as image:
        image.load()
        rgb = ImageOps.exif_transpose(image).convert('RGB')
        h = hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode())
        h.update(rgb.tobytes())
        small = np.asarray(rgb.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
        bits = small[:, 1:] > small[:, :-1]
        dhash = f'{int("".join("1" if b else "0" for b in bits.ravel()), 2):016x}'
        grey = np.asarray(rgb.convert('L').resize((64, 64)), dtype=np.float32)
        return {'readable': True, 'width': width, 'height': height, 'mode': mode,
                'format': fmt, 'pixel_sha256': h.hexdigest(), 'dhash': dhash,
                'brightness': float(grey.mean()),
                'edge_variance': float(np.diff(grey, axis=0).var() + np.diff(grey, axis=1).var())}


def audit(root: Path, mode: str = 'smoke', per_class: int = 5) -> tuple[pd.DataFrame, dict]:
    """Sequentieller Audit; unveraenderte Dateien werden aus SQLite wiederverwendet."""
    if mode not in {'smoke', 'full'} or per_class < 1:
        raise ValueError('mode muss smoke oder full sein; per_class muss positiv sein.')
    p, source = paths(root), inventory(root)
    base = class_root(root)
    selected = source if mode == 'full' else source.groupby('dataset_folder', sort=False).head(per_class)
    policy = digest({'version': AUDIT_VERSION, 'pillow': runtime()['packages']['Pillow']})
    rows, cached = [], 0
    with sqlite3.connect(p['work'] / 'audit_cache.sqlite') as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS audit (path TEXT PRIMARY KEY, sha TEXT, policy TEXT, payload TEXT)')
        for number, item in enumerate(selected.to_dict('records'), 1):
            record = dict(item, readable=False, width=None, height=None, mode='', format='',
                          sha256='', pixel_sha256='', dhash='', brightness=None,
                          edge_variance=None, error='')
            try:
                path = safe_path(base, item['relative_path'])
                stat = path.stat()
                record['file_size_bytes'] = stat.st_size
                record['sha256'] = sha256_file(path)
                old = conn.execute('SELECT sha, policy, payload FROM audit WHERE path=?',
                                   (item['relative_path'],)).fetchone()
                if old and old[0] == record['sha256'] and old[1] == policy:
                    record.update(json.loads(old[2])); cached += 1
                else:
                    record.update(_image_record(path))
                after = path.stat()
                if after.st_size != stat.st_size or after.st_mtime_ns != stat.st_mtime_ns:
                    raise ValueError('Datei wurde waehrend der Pruefung geaendert.')
            except (MemoryError, KeyboardInterrupt):
                raise
            except Exception as exc:
                record['readable'] = False
                record['error'] = f'{type(exc).__name__}: {str(exc)[:250]}'
            rows.append(record)
            if record['readable']:
                payload = {k: record[k] for k in ['readable', 'width', 'height', 'mode', 'format',
                                                 'pixel_sha256', 'dhash', 'brightness', 'edge_variance']}
                conn.execute('INSERT OR REPLACE INTO audit VALUES (?,?,?,?)',
                             (item['relative_path'], record['sha256'], policy, json.dumps(payload)))
            if number % 500 == 0 or number == len(selected):
                conn.commit()
                print(f'Bildpruefung: {number}/{len(selected)} ({cached} aus Cache)', flush=True)
    result = pd.DataFrame(rows)
    for key in ['sha256', 'pixel_sha256']:
        counts = result.loc[result[key].ne(''), key].value_counts()
        result[key + '_group_size'] = result[key].map(counts).fillna(0).astype(int)
    suffix = '_smoke' if mode == 'smoke' else ''
    out = p['manifests'] / f'image_audit{suffix}.csv'
    result.to_csv(out, index=False)
    duplicate = result.sha256_group_size.gt(1)
    cross = result.loc[result.sha256.ne('')].groupby('sha256').dataset_folder.nunique()
    summary = {'audit_mode': mode, 'n_inventory_files': len(source), 'n_files_audited': len(result),
               'n_unreadable_files': int((~result.readable).sum()),
               'n_exact_duplicate_files': int(duplicate.sum()),
               'n_exact_duplicate_groups': int(result.loc[duplicate, 'sha256'].nunique()),
               'n_cross_folder_exact_files': int(result.sha256.map(cross).fillna(0).gt(1).sum()),
               'n_repeated_filenames': int(result.filename.value_counts().gt(1).sum()),
               'inventory_sha256': sha256_file(root / 'data/manifests/file_inventory.csv'),
               'manifest_sha256': sha256_file(out), 'policy': policy, 'runtime': runtime(),
               'raw_data_modified': False}
    write_json(p['work'] / f'02_audit{suffix}.json', summary)
    return result, summary


def load_audit(root: Path) -> pd.DataFrame:
    p = paths(root)
    meta = read_json(p['work'] / '02_audit.json')
    source = root / 'data/manifests/file_inventory.csv'
    file = p['manifests'] / 'image_audit.csv'
    if meta['audit_mode'] != 'full' or sha256_file(source) != meta['inventory_sha256']:
        raise ValueError('Vollstaendiger Audit des aktuellen Inventars erforderlich.')
    if sha256_file(file) != meta['manifest_sha256']:
        raise ValueError('Audit-Datei wurde veraendert.')
    frame = read_csv(file)
    frame['readable'] = as_bool(frame.readable)
    if len(frame) != meta['n_inventory_files']:
        raise ValueError('Audit ist unvollstaendig.')
    return frame


def curate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(frame, ['relative_path', 'sha256', 'pixel_sha256', 'readable', 'dataset_folder'])
    result = frame.copy()
    result['readable'] = as_bool(result.readable)
    result['decision'] = 'KEEP'
    result['canonical_relative_path'] = result.relative_path
    bad = ~result.readable | result.sha256.eq('') | result.pixel_sha256.eq('')
    result.loc[bad, 'decision'] = 'EXCLUDE_UNREADABLE'
    groups = []
    for pixel_hash, group in result.loc[~bad].groupby('pixel_sha256', sort=True):
        if len(group) == 1:
            continue
        ordered = group.sort_values('relative_path')
        canonical = ordered.iloc[0]
        cross = group.dataset_folder.nunique() > 1
        result.loc[group.index, 'canonical_relative_path'] = canonical.relative_path
        if cross:
            result.loc[group.index, 'decision'] = 'REVIEW_CROSS_LABEL_DUPLICATE'
        else:
            result.loc[group.index, 'decision'] = 'EXCLUDE_IDENTICAL_PIXELS'
            same_bytes = group.index[group.sha256.eq(canonical.sha256)]
            result.loc[same_bytes, 'decision'] = 'EXCLUDE_EXACT_DUPLICATE'
            result.loc[canonical.name, 'decision'] = 'KEEP_CANONICAL'
        groups.append({'pixel_sha256': pixel_hash, 'n_files': len(group),
                       'n_classes': group.dataset_folder.nunique(), 'cross_label': cross,
                       'n_file_hashes': group.sha256.nunique(),
                       'dataset_folders': ' | '.join(sorted(group.dataset_folder.unique(), key=folder_key)),
                       'canonical_relative_path': canonical.relative_path})
    result['use_for_model'] = result.decision.isin(['KEEP', 'KEEP_CANONICAL'])
    keep = result.loc[result.use_for_model]
    if keep.pixel_sha256.duplicated().any() or keep.sha256.duplicated().any():
        raise ValueError('Curation hinterlaesst identische freigegebene Inhalte.')
    columns = ['pixel_sha256', 'n_files', 'n_classes', 'cross_label', 'n_file_hashes',
               'dataset_folders', 'canonical_relative_path']
    return result, pd.DataFrame(groups, columns=columns)


def near_candidates(curated: pd.DataFrame) -> pd.DataFrame:
    """Gleicher grober dHash ist nur ein Pruefhinweis, kein Duplikatbeweis."""
    part = curated.loc[curated.use_for_model & curated.dhash.ne('')]
    counts = part.groupby(['dataset_folder', 'dhash']).size().rename('n').reset_index()
    return counts.loc[counts.n.gt(1)].sort_values('n', ascending=False).reset_index(drop=True)


def save_curation(root: Path, frame: pd.DataFrame, groups: pd.DataFrame) -> dict:
    p = paths(root)
    out = p['manifests'] / 'curation_manifest.csv'
    frame.to_csv(out, index=False)
    groups.to_csv(p['manifests'] / 'duplicate_groups.csv', index=False)
    near_candidates(frame).to_csv(p['manifests'] / 'near_duplicate_candidates.csv', index=False)
    summary = {'n_raw_files': len(frame), 'n_candidates': int(frame.use_for_model.sum()),
               'decision_counts': {str(k): int(v) for k, v in frame.decision.value_counts().items()},
               'audit_sha256': sha256_file(p['manifests'] / 'image_audit.csv'),
               'manifest_sha256': sha256_file(out), 'policy': 'exact-bytes-and-oriented-RGB; cross-folder-quarantine',
               'raw_data_modified': False, 'split_created': False}
    write_json(p['work'] / '03_curation.json', summary)
    return summary


def load_curated(root: Path) -> pd.DataFrame:
    p = paths(root)
    state = read_json(p['work'] / '03_curation.json')
    out = p['manifests'] / 'curation_manifest.csv'
    if sha256_file(out) != state['manifest_sha256'] or sha256_file(p['manifests'] / 'image_audit.csv') != state['audit_sha256']:
        raise ValueError('Curation/Audit veraendert. Schritte 02-03 konsistent ausfuehren.')
    frame = read_csv(out)
    return frame.loc[as_bool(frame.use_for_model)].reset_index(drop=True)


def show_examples(root: Path, frame: pd.DataFrame, count: int = 4, title: str = '') -> None:
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display
    base = class_root(root)
    for row in frame.head(count).to_dict('records'):
        path = safe_path(base, row['relative_path'])
        if sha256_file(path) != row['sha256']:
            raise ValueError('Bildinhalt passt nicht zum Audit.')
        character = row.get('character', '')
        display(Markdown(f"**{title}** Ordner `{row['dataset_folder']}` | {character} | "
                         f"{row.get('category', '')}\n\nDatei: `{row['relative_path']}`  \nSHA-256: `{row['sha256']}`"))
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert('RGB')
            fig, ax = plt.subplots(figsize=(5, 3.6))
            ax.imshow(rgb); ax.axis('off'); fig.tight_layout(); plt.show(); plt.close(fig)


def canonical_schema() -> pd.DataFrame:
    vowels = [chr(c) for c in [0xB85,0xB86,0xB87,0xB88,0xB89,0xB8A,0xB8E,0xB8F,0xB90,0xB92,0xB93,0xB94]]
    bases = [chr(c) for c in [0xB95,0xB99,0xB9A,0xB9E,0xB9F,0xBA3,0xBA4,0xBA8,0xBAA,0xBAE,0xBAF,0xBB0,0xBB2,0xBB5,0xBB4,0xBB3,0xBB1,0xBA9]]
    signs = [''] + [chr(c) for c in [0xBBE,0xBBF,0xBC0,0xBC1,0xBC2,0xBC6,0xBC7,0xBC8,0xBCA,0xBCB,0xBCC]]
    rows = [(v, 'uyir', f'V{i:02d}', FIST) for i, v in enumerate(vowels, 1)]
    rows += [(b + chr(0xBCD), 'mei', FIST, f'C{i:02d}') for i, b in enumerate(bases, 1)]
    rows += [(unicodedata.normalize('NFC', b+s), 'uyirmei', f'V{v:02d}', f'C{c:02d}')
             for c, b in enumerate(bases, 1) for v, s in enumerate(signs, 1)]
    rows += [(chr(0xB83), 'ayudha', NA, NA), (BG, 'background', NA, NA)]
    return pd.DataFrame(rows, columns=['character','category','vowel_target','consonant_target'])


def label_review(root: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    """Vorhandene Zuordnungen verwenden; niemals Ordnernummer = Schemaindex annehmen."""
    p = paths(root)
    target = p['config'] / 'label_review.csv'
    legacy = root / 'configs/label_schema_review.csv'
    review = candidates.groupby('dataset_folder').size().rename('n_candidates').reset_index()
    source = target if target.exists() else legacy if legacy.exists() else None
    if source:
        old = read_csv(source)
        require_columns(old, ['dataset_folder', 'character'])
        if old.dataset_folder.duplicated().any():
            raise ValueError('Mehrdeutige Ordnerzuordnung.')
        review = review.merge(old.drop(columns=['n_candidates'], errors='ignore'),
                              on='dataset_folder', how='left', validate='1:1')
    for column, default in [('character',''),('verified',False),('evidence',''),('notes','')]:
        if column not in review:
            review[column] = default
        review[column] = review[column].fillna(default)
    review['character'] = review.character.map(lambda x: unicodedata.normalize('NFC', str(x).strip()))
    exact = set(canonical_schema().character)
    for idx in review.index[review.character.eq('')]:
        folder = review.loc[idx, 'dataset_folder']
        if folder in exact:
            review.loc[idx, 'character'] = folder
        elif folder.casefold() == 'background':
            review.loc[idx, 'character'] = BG
    provisional = review.evidence.str.contains('provisional|pending|ungepr', case=False, regex=True)
    review.loc[provisional, 'verified'] = False
    # Nur die fuer die Pruefung relevanten Spalten; Kategorien werden abgeleitet.
    review = review[['dataset_folder','n_candidates','character','verified','evidence','notes']]
    review = review.sort_values('dataset_folder', key=lambda s:s.map(folder_key)).reset_index(drop=True)
    review.to_csv(target, index=False)
    canonical_schema().to_csv(p['manifests'] / 'canonical_tamil_schema.csv', index=False)
    return review


def validate_labels(review: pd.DataFrame) -> pd.DataFrame:
    review = review.copy()
    review['character'] = review.character.map(lambda x: unicodedata.normalize('NFC', str(x).strip()))
    if not as_bool(review.verified).all() or review.evidence.str.strip().eq('').any():
        raise ValueError('Alle verwendeten Ordnerzuordnungen brauchen verified=True und Quellenbeleg.')
    if review.character.eq('').any() or review.character.duplicated().any():
        raise ValueError('Zeichen fehlen oder sind mehreren Ordnern zugeordnet.')
    schema = review.merge(canonical_schema(), on='character', how='left', validate='1:1')
    if schema.category.isna().any():
        raise ValueError('Unbekanntes Tamil-Zeichen im Labelmapping.')
    return schema


def exposure_hashes(root: Path, files: list[str], audit_frame: pd.DataFrame) -> tuple[set, list]:
    exposed, provenance = set(), []
    for relative in files:
        file = safe_path(root, relative)
        old = read_csv(file)
        require_columns(old, ['sha256'])
        hashes = set(old.sha256) - {''}
        if not hashes or any(len(h) != 64 for h in hashes):
            raise ValueError(f'Kein verlaesslicher Expositionsnachweis: {file}')
        exposed |= hashes
        provenance.append({'path':relative, 'sha256':sha256_file(file), 'n_hashes':len(hashes)})
    unknown = exposed - set(audit_frame.sha256)
    if unknown:
        raise ValueError(f'{len(unknown)} historische Hashwerte fehlen im aktuellen Audit. Datenherkunft pruefen.')
    known_pixels = set(audit_frame.loc[audit_frame.sha256.isin(exposed), 'pixel_sha256']) - {''}
    return known_pixels, provenance


def _rank(key: str, seed: int, phase: str) -> str:
    return hashlib.sha256(f'{seed}:{phase}:{key}'.encode()).hexdigest()


def make_split(frame: pd.DataFrame, excluded_pixels: set, config: dict,
               groups: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ganze Gruppen zuweisen, danach pro Klasse kappen. Keine Signer-IDs erfinden."""
    data = frame.copy().reset_index(drop=True)
    if data.pixel_sha256.duplicated().any():
        raise ValueError('Identische Pixel im Kandidatenbestand.')
    if groups is None:
        data['group_id'] = data.pixel_sha256
        selected = []
        for folder, part in data.groupby('dataset_folder', sort=True):
            category = part.category.iloc[0]
            used = set()
            for phase in ['test', 'validation', 'train']:
                target = int(config['background_eval'] if category == 'background' and phase != 'train'
                             else config['focus_train'] if phase == 'train' and category in ['uyir','mei','ayudha','background']
                             else config[phase])
                allowed = ~part.pixel_sha256.isin(used)
                if phase != 'train':
                    allowed &= ~part.pixel_sha256.isin(excluded_pixels)
                pool = part.loc[allowed].copy()
                if len(pool) < target:
                    raise ValueError(f'Zu wenig Daten fuer {phase}, Ordner {folder}: {len(pool)} < {target}.')
                pool['_rank'] = pool.pixel_sha256.map(lambda h: _rank(h, config['seed'], phase))
                chosen = pool.sort_values('_rank').head(target).drop(columns='_rank')
                used.update(chosen.pixel_sha256)
                selected.append(chosen.assign(split=phase))
        result = pd.concat(selected, ignore_index=True)
        if result.pixel_sha256.duplicated().any():
            raise ValueError('Inhaltsueberschneidung im Split.')
        return result.sort_values(['split','dataset_folder','sha256']).reset_index(drop=True)
    else:
        require_columns(groups, ['relative_path','group_id'])
        if groups.relative_path.duplicated().any():
            raise ValueError('Gruppenmapping ist nicht eindeutig.')
        data = data.merge(groups[['relative_path','group_id']], on='relative_path', how='left', validate='1:1')
        if data.group_id.isna().any() or data.group_id.eq('').any():
            raise ValueError('Verifizierte Gruppen-IDs muessen alle Kandidaten abdecken.')
    by_group = {str(k):list(g.index) for k,g in data.groupby('group_id', sort=False)}
    data['group_id'] = data.group_id.astype(str)
    touched = set(data.loc[data.pixel_sha256.isin(excluded_pixels), 'group_id'])
    counts = data.groupby('dataset_folder').category.first().to_dict()
    unavailable, selected = set(), []
    for phase in ['test', 'validation', 'train']:
        targets = {folder: int(config['background_eval'] if cat == 'background' and phase != 'train'
                               else config['focus_train'] if phase == 'train' and cat in ['uyir','mei','ayudha','background']
                               else config[phase]) for folder,cat in counts.items()}
        available = {g:ix for g,ix in by_group.items()
                     if g not in unavailable and (phase == 'train' or g not in touched)}
        found = {folder:0 for folder in targets}
        chosen = []
        for gid in sorted(available, key=lambda g:_rank(g,config['seed'],phase)):
            part = data.loc[available[gid]]
            if not any(found[c] < targets[c] for c in part.dataset_folder):
                continue
            chosen.append(gid)
            for folder, n in part.dataset_folder.value_counts().items():
                found[folder] += int(n)
            if all(found[c] >= targets[c] for c in targets):
                break
        short = {c:(found[c],targets[c]) for c in targets if found[c] < targets[c]}
        if short:
            raise ValueError(f'Zu wenig unabh. Daten fuer {phase} (vorhanden, Ziel): {short}. '
                             'Umfang vor Freigabe reduzieren oder Gruppenstruktur pruefen.')
        unavailable |= set(chosen)
        pool = data.loc[[ix for g in chosen for ix in available[g]]].copy()
        pool['_rank'] = pool.pixel_sha256.map(lambda h:_rank(h,config['seed'],phase))
        for folder, n in targets.items():
            selected.append(pool.loc[pool.dataset_folder.eq(folder)].sort_values('_rank').head(n).assign(split=phase))
    split = pd.concat(selected, ignore_index=True).drop(columns='_rank')
    if split.groupby('group_id').split.nunique().gt(1).any() or split.pixel_sha256.duplicated().any():
        raise ValueError('Split enthaelt Gruppen- oder Inhaltsueberschneidungen.')
    if set(split.loc[split.split.eq('test'), 'pixel_sha256']) & excluded_pixels:
        raise ValueError('Bereits verwendete Inhalte im Holdout.')
    return split.sort_values(['split','dataset_folder','sha256']).reset_index(drop=True)


def freeze_split(root: Path, config: dict, exposure_files: list[str],
                 exposure_reviewed: bool, review_note: str,
                 groups_file: str | None = None, no_previous_exposure: bool = False) -> tuple[pd.DataFrame,dict]:
    if not exposure_reviewed or len(review_note.strip()) < 20:
        raise ValueError('Datenverwendung und Handstruktur pruefen und kurz dokumentieren.')
    if exposure_files and no_previous_exposure:
        raise ValueError('Historische Manifeste und gleichzeitig leere Datenhistorie sind widerspruechlich.')
    if not exposure_files and not no_previous_exposure:
        raise ValueError('Vorher verwendete Bildmanifeste angeben. Leere Historie nicht stillschweigend annehmen.')
    p = paths(root)
    schema = validate_labels(read_csv(p['config'] / 'label_review.csv'))
    candidates = load_curated(root)
    frame = candidates.merge(schema.drop(columns=['n_candidates']), on='dataset_folder', how='left', validate='m:1')
    if frame.category.isna().any():
        raise ValueError('Labelmapping deckt nicht alle Kandidaten ab.')
    exposed, history = exposure_hashes(root, exposure_files, load_audit(root))
    groups = read_csv(safe_path(root, groups_file)) if groups_file else None
    provenance = {'config':config, 'exposure_files':history, 'exposure_reviewed':True,
                  'no_previous_exposure':no_previous_exposure, 'split_review':review_note,
                  'curation_sha256':sha256_file(p['manifests'] / 'curation_manifest.csv'),
                  'schema_sha256':sha256_file(p['config'] / 'label_review.csv'),
                  'groups_sha256':sha256_file(safe_path(root, groups_file)) if groups_file else None,
                  'split_code_sha256':sha256_file(Path(__file__)),
                  'limitation':'Bildsplit; keine Signer-/Session-Unabhaengigkeit nachgewiesen. '
                               'Aehnliche Aufnahmen koennen verbleiben.' if groups is None
                               else 'Gruppentrennung entsprechend den manuell bestaetigten IDs.'}
    statefile, splitfile = p['work'] / '04_split.json', p['manifests'] / 'split_manifest.csv'
    if statefile.exists():
        state = read_json(statefile)
        if state['input_fingerprint'] != digest(provenance) or sha256_file(splitfile) != state['split_sha256']:
            raise ValueError('Festgeschriebener Split passt nicht zu Eingaben. Nicht ueberschreiben; zuerst archivieren.')
        return read_csv(splitfile), state
    split = make_split(frame, exposed, config, groups)
    split.to_csv(splitfile, index=False)
    schema.to_csv(p['manifests'] / 'label_schema.csv', index=False)
    state = dict(provenance, input_fingerprint=digest(provenance), split_sha256=sha256_file(splitfile),
                 n_selected=len(split), n_previously_used_pixels=len(exposed),
                 split_counts={k:int(v) for k,v in split.split.value_counts().items()})
    write_json(statefile, state)
    return split, state


def load_split(root: Path, require_hand_review: bool = True) -> tuple[pd.DataFrame, dict]:
    p = paths(root)
    state = read_json(p['work'] / '04_split.json')
    file = p['manifests'] / 'split_manifest.csv'
    if sha256_file(file) != state['split_sha256']:
        raise ValueError('Split-Manifeste wurden nach Freigabe veraendert.')
    if require_hand_review:
        hand = read_json(p['work'] / '04_hand_review.json')
        if not hand.get('approved') or hand['split_sha256'] != state['split_sha256']:
            raise ValueError('Handstruktur ist fuer diesen Split noch nicht freigegeben.')
    return read_csv(file), state


def approve_hand_structure(root: Path, approved: bool, note: str) -> None:
    if not approved or len(note.strip()) < 30 or '...' in note:
        raise ValueError('Beobachtungen zu Uyir, Mei, Uyir-Mei, Ayudha und Background dokumentieren.')
    _, state = load_split(root, require_hand_review=False)
    write_json(paths(root)['work'] / '04_hand_review.json',
               {'approved': True, 'note': note, 'split_sha256': state['split_sha256'],
                'convention': 'FIST is an inactive component label, not a detector failure; Ayudha separate'})
