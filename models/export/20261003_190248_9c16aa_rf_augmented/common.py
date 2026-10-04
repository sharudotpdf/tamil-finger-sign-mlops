"""Pfade, Fingerprints und kleine Ein-/Ausgabefunktionen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False,
                       dtype={'dataset_folder': str, 'sha256': str,
                              'pixel_sha256': str, 'dhash': str})


def as_bool(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    if not text.isin(['true', 'false', '1', '0']).all():
        raise ValueError('Ungueltige boolesche Werte: ' + str(text.unique()))
    return text.isin(['true', '1'])


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'Fehlende Spalten: {sorted(missing)}')


def safe_path(base: Path, relative: str) -> Path:
    rel = PurePosixPath(str(relative))
    if rel.is_absolute() or '..' in rel.parts or '\\' in str(relative):
        raise ValueError(f'Ungueltiger relativer Pfad: {relative}')
    path = base.joinpath(*rel.parts).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f'Pfad verlaesst Datenverzeichnis: {relative}')
    return path


def paths(root: Path) -> dict[str, Path]:
    root = Path(root).resolve()
    p = {'root': root, 'manifests': root / 'data/manifests/tlfs23',
         'work': root / 'artifacts/tlfs23', 'config': root / 'configs/tlfs23'}
    for name in ('manifests', 'work', 'config'):
        p[name].mkdir(parents=True, exist_ok=True)
    return p


def class_root(root: Path) -> Path:
    summary = read_json(root / 'artifacts/data_audit/01_inventory_summary.json')
    relative = summary.get('class_root_relative')
    base = safe_path(root, relative) if relative else Path(summary['class_root'])
    if not base.is_dir():
        raise FileNotFoundError(f'Klassenordner fehlt: {base}. Notebook 01 ausfuehren.')
    return base


def table_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    stable = frame[columns].astype(str).sort_values(columns).reset_index(drop=True)
    return hashlib.sha256(stable.to_csv(index=False, lineterminator='\n').encode()).hexdigest()


def folder_key(value: str) -> tuple:
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def runtime() -> dict:
    import platform
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for name in ['numpy', 'pandas', 'Pillow', 'mediapipe', 'scikit-learn']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'architecture': platform.machine(),
            'system': platform.system(), 'packages': packages}
