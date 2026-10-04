"""Bildpruefung ohne Aenderung von Labels, Rohdaten oder Curation-Regeln."""
from __future__ import annotations

import base64
from html import escape
from io import BytesIO
from pathlib import Path
import unicodedata

import pandas as pd
from PIL import Image, ImageOps

from . import data as d
from .common import (as_bool, class_root, digest, folder_key, paths, read_csv,
                     read_json, require_columns, safe_path, sha256_file, write_json)

CATEGORIES = ['uyir', 'mei', 'uyirmei', 'ayudha', 'background']
EXPOSURE_NAME = 'visual_review_exposure.csv'


def _text(value) -> str:
    return '' if pd.isna(value) else str(value)


def _save_csv(file: Path, frame: pd.DataFrame) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix('.csv.tmp')
    frame.to_csv(temp, index=False, lineterminator='\n')
    temp.replace(file)


def _split(root: Path):
    if (paths(root)['work'] / '04_split.json').is_file():
        return d.load_split(root, require_hand_review=False)[0]
    return None


def review_pool(root: Path, frame: pd.DataFrame, train_only: bool = False) -> pd.DataFrame:
    """Nach Split-Freigabe niemals Validation/Holdout als Review-Bilder anzeigen."""
    plan = _split(root)
    if plan is None:
        return frame.copy()
    if train_only:
        allowed = set(plan.loc[plan.split.eq('train'), 'pixel_sha256'])
        return frame.loc[frame.pixel_sha256.isin(allowed)].copy()
    blocked = set(plan.loc[~plan.split.eq('train'), 'pixel_sha256'])
    return frame.loc[~frame.pixel_sha256.isin(blocked)].copy()


def _record_exposure(root: Path, frame: pd.DataFrame, context: str) -> None:
    """Vor Split-Auswahl angebotene Bilder werden konservativ als gesehen erfasst."""
    require_columns(frame, ['relative_path', 'sha256', 'pixel_sha256'])
    plan = _split(root)
    if plan is not None:
        blocked = set(plan.loc[~plan.split.eq('train'), 'pixel_sha256'])
        if set(frame.pixel_sha256) & blocked:
            raise ValueError('Review wuerde Validation/Holdout zeigen. Auswahl nicht ausgefuehrt.')
        # Nur neue, noch nicht zugeteilte Bilder muessen vor kuenftigen Splits geschuetzt werden.
        frame = frame.loc[~frame.pixel_sha256.isin(set(plan.pixel_sha256))]
    if frame.empty:
        return
    file = paths(root)['manifests'] / EXPOSURE_NAME
    rows = frame[['relative_path', 'sha256', 'pixel_sha256']].assign(review_context=context)
    old = read_csv(file) if file.exists() else rows.iloc[:0]
    merged = pd.concat([old, rows], ignore_index=True)
    merged = merged.drop_duplicates(['relative_path', 'sha256', 'review_context'])
    merged = merged.sort_values(['relative_path', 'sha256', 'review_context']).reset_index(drop=True)
    if not file.exists() or not merged.equals(old):
        _save_csv(file, merged)


def _html_image(root: Path, row: dict) -> str:
    path = safe_path(class_root(root), row['relative_path'])
    if sha256_file(path) != row['sha256']:
        raise ValueError('Bildinhalt weicht vom Audit ab: ' + row['relative_path'])
    with Image.open(path) as image:
        image.load()
        rgb = ImageOps.exif_transpose(image).convert('RGB')
        rgb.thumbnail((420, 320), Image.Resampling.LANCZOS)
        stream = BytesIO()
        rgb.save(stream, format='PNG')
    encoded = base64.b64encode(stream.getvalue()).decode('ascii')
    fields = [
        ('Ordner', row.get('dataset_folder', '')),
        ('Zeichen laut Mapping', _text(row.get('character')) or 'NICHT ZUGEORDNET'),
        ('Kategorie', _text(row.get('category')) or 'NICHT ZUGEORDNET'),
        ('Vokalkomponente', row.get('vowel_display', '')),
        ('Konsonantenkomponente', row.get('consonant_display', '')),
        ('Mappingstatus', row.get('mapping_status', '')),
        ('Datei', row['relative_path']),
        ('dHash', row.get('dhash', '')),
    ]
    details = ''.join('<div><b>' + escape(label) + ':</b> ' + escape(_text(value)) + '</div>'
                      for label, value in fields if _text(value))
    for label, key in [('Datei-SHA-256', 'sha256'), ('Pixel-SHA-256', 'pixel_sha256')]:
        if _text(row.get(key)):
            details += '<details><summary>' + label + '</summary><code style="overflow-wrap:anywhere">' \
                       + escape(_text(row[key])) + '</code></details>'
    return ('<section style="flex:1 1 280px;max-width:450px;padding:10px;'
            'border:1px solid #bbb;border-radius:5px">'
            '<img alt="' + escape(row['relative_path'], quote=True) + '" '
            'style="width:100%;max-height:320px;object-fit:contain" src="data:image/png;base64,'
            + encoded + '"><div style="overflow-wrap:anywhere;line-height:1.45">'
            + details + '</div></section>')


def show_cards(root: Path, frame: pd.DataFrame, title: str = '',
               context: str = 'review', output_name: str | None = None) -> pd.DataFrame:
    """Kleine HTML-Galerie; Tamil-Text nutzt die Schriftdarstellung von VS Code."""
    from IPython.display import HTML, display
    if frame.empty:
        print('Keine Beispiele in dieser Auswahl. Filter bzw. Seitenzahl pruefen.')
        return frame.copy()
    if len(frame) > 32:
        raise ValueError('Maximal 32 Bilder je Galerie. Auswahl verkleinern.')
    require_columns(frame, ['relative_path', 'sha256', 'pixel_sha256', 'dataset_folder'])
    _record_exposure(root, frame, context)
    cards = []
    for row in frame.to_dict('records'):
        try:
            cards.append(_html_image(root, row))
        except (OSError, ValueError) as exc:
            raise ValueError(f'Bildpruefung abgebrochen: {row["relative_path"]}: {exc}') from exc
    html = ('<div style="font-family:system-ui,\'Tamil Sangam MN\',\'Noto Sans Tamil\',sans-serif">'
            '<h4>' + escape(title) + '</h4><div style="display:flex;flex-wrap:wrap;gap:12px">'
            + ''.join(cards) + '</div></div>')
    display(HTML(html))
    if output_name:
        if Path(output_name).name != output_name:
            raise ValueError('Nur einen Dateinamen fuer die Galerie angeben.')
        directory = paths(root)['work'] / 'reviews'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (output_name + '.html')).write_text(
            '<!doctype html><meta charset="utf-8">' + html, encoding='utf-8')
        _save_csv(directory / (output_name + '.csv'), frame)
    return frame.copy()


def dhash_pairs(root: Path, curated: pd.DataFrame, max_pairs: int = 6,
                offset: int = 0) -> pd.DataFrame:
    """Wie near_candidates: gleicher dHash innerhalb desselben Ordners, Distanz 0.

    Ein Paar pro Gruppe, nur unterschiedliche Pixelinhalte. Keine umfassende
    Suche ueber positive Hamming-Distanzen und keine automatische Curation.
    """
    if not 1 <= max_pairs <= 12 or offset < 0:
        raise ValueError('max_pairs: 1 bis 12, offset: mindestens 0.')
    require_columns(curated, ['use_for_model', 'dhash', 'pixel_sha256', 'sha256',
                             'relative_path', 'dataset_folder'])
    pool = review_pool(root, curated)
    valid = as_bool(pool.use_for_model) & pool.dhash.astype(str).str.fullmatch(r'[0-9a-fA-F]{16}', na=False)
    pool = pool.loc[valid].copy()
    pool['use_for_model'] = as_bool(pool.use_for_model)
    pool['dhash'] = pool.dhash.str.lower()
    groups = d.near_candidates(pool).sort_values(
        ['n', 'dataset_folder', 'dhash'], ascending=[False, True, True])
    lookup = pool.groupby(['dataset_folder', 'dhash'], sort=False)
    records = []
    for group in groups.iloc[offset:offset + max_pairs].itertuples():
        examples = lookup.get_group((group.dataset_folder, group.dhash))
        examples = examples.sort_values('relative_path').drop_duplicates('pixel_sha256').head(2)
        if len(examples) != 2:
            continue
        a, b = examples.to_dict('records')
        item = {'pair_id': digest([a['relative_path'], a['sha256'], b['relative_path'], b['sha256']])[:16],
                'dataset_folder': group.dataset_folder, 'dhash': group.dhash,
                'hamming_distance': (int(a['dhash'], 16) ^ int(b['dhash'], 16)).bit_count(),
                'group_size': int(group.n), 'pixel_identical': a['pixel_sha256'] == b['pixel_sha256']}
        for suffix, row in [('a', a), ('b', b)]:
            for field in ['relative_path', 'sha256', 'pixel_sha256', 'dataset_folder', 'dhash']:
                item[field + '_' + suffix] = row[field]
        records.append(item)
    columns = ['pair_id', 'dataset_folder', 'dhash', 'hamming_distance', 'group_size', 'pixel_identical']
    columns += [field + '_' + suffix for suffix in ['a', 'b']
                for field in ['relative_path', 'sha256', 'pixel_sha256', 'dataset_folder', 'dhash']]
    return pd.DataFrame(records, columns=columns)


def show_pairs(root: Path, pairs: pd.DataFrame) -> None:
    if pairs.empty:
        print('Keine passenden dHash-Paare in dieser Auswahl; keine weitere Freigabe erforderlich.')
        return
    for row in pairs.to_dict('records'):
        examples = pd.DataFrame([{field: row[field + '_' + suffix]
                                 for field in ['relative_path', 'sha256', 'pixel_sha256', 'dataset_folder', 'dhash']}
                                for suffix in ['a', 'b']])
        title = (f"Paar {row['pair_id']} | gleicher dHash | Distanz {row['hamming_distance']} "
                 f"| unterschiedliche Pixelinhalte | Gruppe: {row['group_size']} Bilder")
        show_cards(root, examples, title, '03_dhash', '03_dhash_' + row['pair_id'])


def save_dhash_notes(root: Path, pairs: pd.DataFrame, notes: dict | None = None) -> pd.DataFrame:
    """Nur ausdrueckliche Eintraege aendern; fruehere Notizen bleiben erhalten."""
    file = paths(root)['work'] / 'reviews' / '03_dhash_review.csv'
    old = read_csv(file) if file.exists() else pd.DataFrame()
    fresh = pairs.assign(assessment='OFFEN', note='')
    result = pd.concat([old, fresh], ignore_index=True).drop_duplicates('pair_id', keep='first')
    for pair_id, item in (notes or {}).items():
        if pair_id not in set(result.pair_id):
            raise ValueError('Unbekannte Paar-ID: ' + str(pair_id))
        if not isinstance(item, dict) or item.get('assessment') not in {
                'OFFEN', 'SEHR_AEHNLICH', 'UNTERSCHIEDLICH', 'UNKLAR'}:
            raise ValueError('Bewertung: OFFEN, SEHR_AEHNLICH, UNTERSCHIEDLICH oder UNKLAR.')
        mask = result.pair_id.eq(pair_id)
        result.loc[mask, 'assessment'] = item['assessment']
        result.loc[mask, 'note'] = str(item.get('note', ''))
    _save_csv(file, result)
    return result


def label_view(review: pd.DataFrame) -> pd.DataFrame:
    require_columns(review, ['dataset_folder', 'character', 'verified', 'evidence'])
    if review.dataset_folder.duplicated().any():
        raise ValueError('Mehrere Eintraege fuer denselben Ordner.')
    frame = review.copy()
    frame['dataset_folder'] = frame.dataset_folder.astype(str)
    frame['character'] = frame.character.map(lambda x: unicodedata.normalize('NFC', _text(x).strip()))
    canonical = d.canonical_schema()
    frame = frame.drop(columns=['category','vowel_target','consonant_target'], errors='ignore')
    frame = frame.merge(canonical, on='character', how='left', validate='m:1')
    frame['mapping_status'] = as_bool(frame.verified).map({True:'zuvor bestaetigt', False:'noch ungeprueft'})
    frame.loc[frame.category.isna(), 'mapping_status'] = 'NICHT ZUGEORDNET / UNBEKANNT'
    for prefix, category in [('vowel','uyir'), ('consonant','mei')]:
        lookup = canonical.loc[canonical.category.eq(category)].set_index(prefix + '_target').character.to_dict()
        lookup.update({d.FIST:'FIST (Faust)', d.NA:'-- (Sonderfall)'})
        frame[prefix + '_display'] = frame[prefix + '_target'].map(lookup).fillna('NICHT ZUGEORDNET')
    return frame


def apply_mapping_updates(root: Path, review: pd.DataFrame, updates: dict) -> pd.DataFrame:
    result = review.copy()
    for folder, character in updates.items():
        mask = result.dataset_folder.eq(str(folder))
        if int(mask.sum()) != 1:
            raise ValueError(f'Unbekannter/mehrdeutiger Ordner: {folder}')
        character = unicodedata.normalize('NFC', str(character).strip())
        if result.loc[mask, 'character'].iloc[0] != character:
            result.loc[mask, 'character'] = character
            result.loc[mask, 'verified'] = False
            result.loc[mask, 'evidence'] = ''
    file = paths(root)['config'] / 'label_review.csv'
    payload = result.to_csv(index=False, lineterminator='\n')
    if file.exists() and file.read_text(encoding='utf-8') != payload:
        if (paths(root)['work'] / '04_split.json').exists():
            raise ValueError('Mappingaenderung bei festem Split: zuerst getrennten Arbeitsstand planen; nichts ersetzt.')
        backup = paths(root)['work'] / 'reviews' / ('label_review_' + sha256_file(file)[:12] + '.csv')
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_bytes(file.read_bytes())
    _save_csv(file, result)
    return result


def folder_examples(root: Path, candidates: pd.DataFrame, view: pd.DataFrame,
                    folders: list[str] | None = None, page: int | None = None,
                    page_size: int = 8, per_folder: int = 2) -> pd.DataFrame:
    if per_folder < 1 or page_size < 1 or page_size * per_folder > 32:
        raise ValueError('Eine Galerie soll hoechstens 32 Bilder enthalten.')
    if folders and page is not None:
        raise ValueError('Entweder Ordnerliste oder Seite waehlen, nicht beides.')
    if folders:
        chosen = [str(f) for f in folders]
    elif page is not None:
        if page < 0:
            raise ValueError('Seitenzahl darf nicht negativ sein.')
        ordered = sorted(view.dataset_folder, key=folder_key)
        chosen = ordered[page * page_size:(page + 1) * page_size]
    else:
        chosen = []
        for category in CATEGORIES:
            group = view.loc[view.category.eq(category)].sort_values('dataset_folder', key=lambda s:s.map(folder_key))
            chosen.extend(group.dataset_folder.head(2 if category in CATEGORIES[:3] else 1))
        if not chosen:
            chosen = sorted(view.dataset_folder, key=folder_key)[:page_size]
    unknown = set(chosen) - set(view.dataset_folder)
    if unknown:
        raise ValueError(f'Unbekannte Ordner: {sorted(unknown)}')
    if len(chosen) * per_folder > 32:
        raise ValueError('Zu viele Bilder auf einmal. Ordnerliste verkleinern.')
    pool = review_pool(root, candidates, train_only=True)
    if pool.empty:
        return pool.merge(view.drop(columns='n_candidates', errors='ignore'), on='dataset_folder', how='left')
    pieces = [pool.loc[pool.dataset_folder.eq(folder)].sort_values(['sha256','relative_path']).head(per_folder)
              for folder in chosen]
    selected = pd.concat(pieces, ignore_index=True) if pieces else pool.iloc[:0]
    metadata = ['dataset_folder','character','category','vowel_target','consonant_target',
                'vowel_display','consonant_display','mapping_status']
    return selected.drop(columns=metadata[1:], errors='ignore').merge(
        view[metadata], on='dataset_folder', how='left', validate='m:1')


def approve_mapping(root: Path, review: pd.DataFrame, evidence: str) -> pd.DataFrame:
    if len(evidence.strip()) < 15:
        raise ValueError('Tatsaechlich verwendete Referenz und Umfang der Mappingpruefung eintragen.')
    approved = review.copy()
    approved['verified'] = True
    blank = approved.evidence.fillna('').str.strip().eq('')
    provisional = approved.evidence.fillna('').str.contains('provisional|pending|ungepr', case=False, regex=True)
    approved.loc[blank | provisional, 'evidence'] = evidence.strip()
    schema = d.validate_labels(approved)
    p = paths(root)
    file = p['config'] / 'label_review.csv'
    payload = approved.to_csv(index=False, lineterminator='\n').encode('utf-8')
    if (p['work'] / '04_split.json').exists():
        import hashlib
        if hashlib.sha256(payload).hexdigest() != read_json(p['work'] / '04_split.json')['schema_sha256']:
            raise ValueError('Freigabe wuerde festgeschriebenes Mapping aendern. Split bleibt unveraendert.')
    _save_csv(file, approved)
    write_json(p['work'] / 'reviews' / '04_mapping_review.json', {
        'confirmed': True, 'evidence': evidence.strip(), 'schema_sha256': sha256_file(file),
        'scope': 'Quellenabgleich aller verwendeten Labels; Bilder nur als Sichtprobe'})
    return schema


def _exposure_snapshot(root: Path) -> str | None:
    p = paths(root)
    file = p['manifests'] / EXPOSURE_NAME
    if not file.exists():
        return None
    frame = read_csv(file)[['relative_path','sha256','pixel_sha256']]
    frame = frame.drop_duplicates().sort_values(['relative_path','sha256']).reset_index(drop=True)
    if frame.empty:
        return None
    key = digest(frame.to_dict('records'))[:16]
    snapshot = p['manifests'] / ('review_exposure_' + key + '.csv')
    if not snapshot.exists():
        _save_csv(snapshot, frame)
    elif read_csv(snapshot).to_dict('records') != frame.to_dict('records'):
        raise ValueError('Review-Snapshot passt nicht mehr zu seinem Inhalt.')
    return snapshot.relative_to(root).as_posix()


def freeze_or_resume(root: Path, config: dict, exposure_files: list[str], reviewed: bool,
                     note: str, groups_file: str | None = None,
                     no_previous_exposure: bool = False):
    """Ergaenzt Sichtproben-Nachweis; vorhandene Splits werden nicht neu gezogen."""
    if not reviewed or len(note.strip()) < 20:
        raise ValueError('Datenhistorie pruefen: EXPOSURE_REVIEWED und SPLIT_NOTE fehlen.')
    if exposure_files and no_previous_exposure:
        raise ValueError('Historische Dateien und NO_PREVIOUS_EXPOSURE=True widersprechen sich.')
    if not exposure_files and not no_previous_exposure:
        raise ValueError('Vorherige Datenverwendung angeben; leere Historie nicht annehmen.')
    p = paths(root)
    statefile = p['work'] / '04_split.json'
    if statefile.exists():
        state = read_json(statefile)
        stored = [item['path'] for item in state['exposure_files']]
        requested = set(exposure_files)
        historical = {name for name in stored if not Path(name).name.startswith('review_exposure_')}
        if config != state['config'] or requested != historical:
            raise ValueError('Konfiguration/Datenhistorie weicht vom festen Split ab. Nicht automatisch neu aufteilen.')
        plan, current = d.freeze_split(root, config, stored, True, state['split_review'],
            groups_file=groups_file, no_previous_exposure=state.get('no_previous_exposure', False))
        ledger = p['manifests'] / EXPOSURE_NAME
        if ledger.exists():
            exposed = set(read_csv(ledger).pixel_sha256)
            held = set(plan.loc[~plan.split.eq('train'), 'pixel_sha256'])
            if exposed & held:
                raise ValueError('Sichtproben ueberschneiden sich mit Validation/Holdout. Nicht als unberuehrt freigeben.')
        print('Vorhandenen Split geladen; keine neue Zufallsauswahl.')
        return plan, current
    combined = list(exposure_files)
    snapshot = _exposure_snapshot(root)
    if snapshot and snapshot not in combined:
        combined.append(snapshot)
    return d.freeze_split(root, config, combined, True, note, groups_file=groups_file,
                          no_previous_exposure=no_previous_exposure and not combined)


def hand_examples(plan: pd.DataFrame, view: pd.DataFrame, per_category: int = 3) -> pd.DataFrame:
    if not 1 <= per_category <= 6:
        raise ValueError('Ein bis sechs Bilder je Kategorie waehlen.')
    train = plan.loc[plan.split.eq('train')].copy()
    pieces = []
    for category in CATEGORIES:
        pool = train.loc[train.category.eq(category)].sort_values(['sha256','relative_path']).copy()
        pool['_round'] = pool.groupby('dataset_folder').cumcount()
        chosen = pool.sort_values(['_round','sha256']).head(per_category).drop(columns='_round')
        pieces.append(chosen)
    selected = pd.concat(pieces, ignore_index=True)
    return selected.drop(columns=['vowel_display','consonant_display','mapping_status'], errors='ignore').merge(
        view[['dataset_folder','vowel_display','consonant_display','mapping_status']],
        on='dataset_folder', how='left', validate='m:1')


def approve_hands(root: Path, examples: pd.DataFrame, note: str) -> None:
    if examples.empty or set(examples.category) != set(CATEGORIES):
        raise ValueError('Sichtprobe muss Uyir, Mei, Uyir-Mei, Ayudha und Background abdecken.')
    if not examples.split.eq('train').all():
        raise ValueError('Handstruktur nur anhand von Trainingsbildern pruefen.')
    d.approve_hand_structure(root, True, note)
    report = paths(root)['work'] / 'reviews' / '04_hand_examples.csv'
    _save_csv(report, examples)
    evidence = paths(root)['work'] / '04_hand_review.json'
    value = read_json(evidence)
    value['visual_examples'] = report.relative_to(root).as_posix()
    value['visual_examples_sha256'] = sha256_file(report)
    write_json(evidence, value)
