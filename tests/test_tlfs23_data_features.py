from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tlfs23 import data as d, features as f
from tlfs23.common import as_bool, paths, read_csv, read_json, safe_path, sha256_file, write_json


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    for subdir in [
        'notebooks',
        'data/manifests',
        'artifacts/data_audit',
        'models/detector',
        'configs',
    ]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    base = root / 'data/raw/TLFS23 - Tamil Language Finger Spelling Image Dataset 2/Dataset Folders'
    base.mkdir(parents=True, exist_ok=True)
    write_json(
        root / 'artifacts/data_audit/01_inventory_summary.json',
        {
            'class_root_relative': base.relative_to(root).as_posix(),
            'class_root': str(base),
        },
    )
    return root, base


def add_image(base: Path, folder: str, name: str, seed: int = 1, metadata: str | None = None) -> Path:
    path = base / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.random.default_rng(seed).integers(0, 255, (24, 32, 3), dtype=np.uint8)
    image = Image.fromarray(rgb)
    if metadata:
        info = PngImagePlugin.PngInfo()
        info.add_text('note', metadata)
        image.save(path, pnginfo=info)
    else:
        image.save(path)
    return path


def make_inventory(root: Path, base: Path) -> None:
    rows = []
    for path in sorted(base.rglob('*')):
        if path.is_file():
            rows.append(
                {
                    'relative_path': path.relative_to(base).as_posix(),
                    'dataset_folder': path.parent.name,
                    'filename': path.name,
                    'extension': path.suffix,
                    'file_size_bytes': path.stat().st_size,
                }
            )
    pd.DataFrame(rows).to_csv(root / 'data/manifests/file_inventory.csv', index=False)


def prepare_split_project(project, n_per_class: int = 12):
    root, base = project
    schema = d.canonical_schema()
    characters = [schema.iloc[i].character for i in [0, 12, 30, 246, 247]]
    folders = ['1', '13', '31', '247', 'Background']

    for folder_index, folder in enumerate(folders):
        for image_index in range(n_per_class):
            add_image(base, folder, f'{image_index:03d}.png', folder_index * 1000 + image_index)

    make_inventory(root, base)
    audit, _ = d.audit(root, 'full')
    curated, groups = d.curate(audit)
    d.save_curation(root, curated, groups)

    p = paths(root)
    review = pd.DataFrame(
        {
            'dataset_folder': folders,
            'n_candidates': [n_per_class] * len(folders),
            'character': characters,
            'verified': True,
            'evidence': ['Synthetic fixture only'] * len(folders),
            'notes': [''] * len(folders),
        }
    )
    review.to_csv(p['config'] / 'label_review.csv', index=False)

    config = {
        'seed': 42,
        'train': 3,
        'focus_train': 3,
        'validation': 2,
        'test': 2,
        'background_eval': 2,
    }
    plan, state = d.freeze_split(
        root,
        config,
        [],
        True,
        'Synthetic test fixture with no previous data exposure.',
        no_previous_exposure=True,
    )
    d.approve_hand_structure(
        root,
        True,
        'Synthetic hand review covers all five categories for testing only.',
    )
    return root, base, plan, state, config


def landmarks(offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(11)
    hand = rng.uniform(0.1, 0.3, (21, 3)).astype(np.float32)
    hand[:, 0] += offset
    return hand


class FakeDetector:
    def __init__(self, counts=(2,)):
        self.counts = list(counts)
        self.calls = 0

    def detect(self, image):
        count = self.counts[min(self.calls, len(self.counts) - 1)]
        self.calls += 1
        return [landmarks(i * 0.45) for i in range(count)]


def fake_launch(root, job_file, log, timeout):
    job = read_json(job_file)
    log.write_text('SIMULATED DETECTION\n')
    if job.get('preflight'):
        return
    frame = pd.DataFrame(job['rows'])
    rows, features, lm = f.process_batch(
        Path(job['base']), frame, FakeDetector(), job['config']
    )
    f._save_chunk(Path(job['output']), rows, features, lm)


CONFIG = {
    'detection_threshold': 0.5,
    'presence_threshold': 0.5,
    'retry': False,
    'max_side': 768,
}


# ---------- common.py ----------

def test_boolean_parsing_and_invalid_values():
    assert as_bool(pd.Series([True, False, '1', '0'])).tolist() == [True, False, True, False]
    with pytest.raises(ValueError):
        as_bool(pd.Series(['maybe']))


def test_safe_path_blocks_escape_and_symlink(tmp_path):
    for relative in ['../secret', '/etc/passwd', 'a/../../x', 'a\\b']:
        with pytest.raises(ValueError):
            safe_path(tmp_path, relative)

    base = tmp_path / 'base'
    outside = tmp_path / 'outside'
    base.mkdir()
    outside.mkdir()
    (base / 'link').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        safe_path(base, 'link/file.txt')


def test_sha256_file(tmp_path):
    path = tmp_path / 'payload.bin'
    path.write_bytes(b'abc')
    assert sha256_file(path) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'


# ---------- data.py ----------

def test_audit_full_uses_cache_and_detects_inventory_change(project, monkeypatch):
    root, base = project
    for folder in ['1', '2']:
        for i in range(3):
            add_image(base, folder, f'{i}.png', int(folder) * 10 + i)
    make_inventory(root, base)

    smoke, meta = d.audit(root, 'smoke', 1)
    assert len(smoke) == 2
    assert meta['audit_mode'] == 'smoke'
    with pytest.raises(FileNotFoundError):
        d.load_audit(root)

    full, _ = d.audit(root, 'full')
    assert len(full) == 6
    assert full.readable.all()

    def should_not_decode(_):
        raise AssertionError('Unchanged image should come from cache')

    monkeypatch.setattr(d, '_image_record', should_not_decode)
    cached, _ = d.audit(root, 'full')
    assert cached.readable.all()

    inventory = root / 'data/manifests/file_inventory.csv'
    inventory.write_text(inventory.read_text() + '\n')
    with pytest.raises(ValueError):
        d.load_audit(root)


def test_curation_handles_exact_pixel_and_cross_folder_duplicates(project):
    root, base = project
    source = add_image(base, '1', 'a.png', 7)
    shutil.copy(source, base / '1/b.png')
    add_image(base, '1', 'metadata.png', 7, metadata='different bytes, same pixels')
    add_image(base, '2', 'cross.png', 7)
    add_image(base, '2', 'unique.png', 8)
    make_inventory(root, base)

    audit, _ = d.audit(root, 'full')
    curated, groups = d.curate(audit)

    cross_rows = curated.loc[curated.dataset_folder.isin(['1', '2']) & curated.pixel_sha256.eq(audit.pixel_sha256.iloc[0])]
    assert cross_rows.decision.eq('REVIEW_CROSS_LABEL_DUPLICATE').all()
    assert groups.cross_label.any()
    assert curated.use_for_model.sum() == 1


def test_curation_excludes_corrupt_or_missing_files(project):
    root, base = project
    add_image(base, '1', 'good.png', 1)
    (base / '1/corrupt.png').write_bytes(b'not an image')
    missing = base / '1/missing.png'
    missing.write_bytes(b'temporary')
    make_inventory(root, base)
    missing.unlink()

    audit, meta = d.audit(root, 'full')
    curated, _ = d.curate(audit)
    assert meta['n_unreadable_files'] == 2
    assert curated.decision.eq('EXCLUDE_UNREADABLE').sum() == 2


def test_canonical_schema_has_expected_248_classes():
    schema = d.canonical_schema()
    assert len(schema) == 248
    assert schema.character.nunique() == 248
    assert schema.category.value_counts().to_dict() == {
        'uyirmei': 216,
        'mei': 18,
        'uyir': 12,
        'ayudha': 1,
        'background': 1,
    }
    assert schema.loc[schema.category.eq('mei'), 'vowel_target'].eq('FIST').all()
    assert schema.loc[schema.category.eq('uyir'), 'consonant_target'].eq('FIST').all()


def test_label_review_does_not_guess_numeric_folder_mapping(project):
    root, _ = project
    review = d.label_review(root, pd.DataFrame({'dataset_folder': ['1', '2', 'Background']}))
    assert review.loc[review.dataset_folder.isin(['1', '2']), 'character'].eq('').all()
    assert not as_bool(review.verified).any()


def test_provisional_mapping_cannot_be_verified_implicitly(project):
    root, _ = project
    p = paths(root)
    pd.DataFrame(
        {
            'dataset_folder': ['1'],
            'character': ['அ'],
            'verified': [True],
            'evidence': ['PROVISIONAL mapping'],
            'notes': [''],
        }
    ).to_csv(p['config'] / 'label_review.csv', index=False)

    review = d.label_review(root, pd.DataFrame({'dataset_folder': ['1']}))
    assert not as_bool(review.verified).any()
    with pytest.raises(ValueError):
        d.validate_labels(review)


def test_split_is_deterministic_disjoint_and_loadable(project):
    root, _, plan, state, config = prepare_split_project(project)
    assert not plan.pixel_sha256.duplicated().any()
    assert plan.groupby('group_id').split.nunique().max() == 1
    assert {'train', 'validation', 'test'} <= set(plan.split)

    loaded, loaded_state = d.load_split(root)
    pd.testing.assert_frame_equal(loaded, read_csv(paths(root)['manifests'] / 'split_manifest.csv'))
    assert loaded_state == state

    again, state_again = d.freeze_split(
        root,
        config,
        [],
        True,
        'Synthetic test fixture with no previous data exposure.',
        no_previous_exposure=True,
    )
    assert state_again == state
    assert again.sha256.tolist() == plan.sha256.tolist()


def test_frozen_split_blocks_changed_configuration(project):
    root, _, _, _, config = prepare_split_project(project)
    changed = dict(config, seed=99)
    with pytest.raises(ValueError):
        d.freeze_split(
            root,
            changed,
            [],
            True,
            'Synthetic test fixture with no previous data exposure.',
            no_previous_exposure=True,
        )


def test_exposed_pixels_are_excluded_from_validation_and_test(project):
    root, base = project
    schema = d.canonical_schema()
    folders = ['1', '13', '31', '247', 'Background']
    characters = [schema.iloc[i].character for i in [0, 12, 30, 246, 247]]
    for folder_index, folder in enumerate(folders):
        for image_index in range(12):
            add_image(base, folder, f'{image_index:03d}.png', folder_index * 1000 + image_index)
    make_inventory(root, base)
    audit, _ = d.audit(root, 'full')
    curated, groups = d.curate(audit)
    d.save_curation(root, curated, groups)

    p = paths(root)
    pd.DataFrame(
        {
            'dataset_folder': folders,
            'n_candidates': [12] * 5,
            'character': characters,
            'verified': True,
            'evidence': 'Synthetic fixture only',
            'notes': '',
        }
    ).to_csv(p['config'] / 'label_review.csv', index=False)

    exposed_row = audit.iloc[[0]][['sha256']]
    exposure_file = root / 'data/manifests/previous_use.csv'
    exposed_row.to_csv(exposure_file, index=False)
    exposed_pixel = audit.loc[audit.sha256.eq(exposed_row.sha256.iloc[0]), 'pixel_sha256'].iloc[0]

    config = {'seed': 42, 'train': 3, 'focus_train': 3, 'validation': 2, 'test': 2, 'background_eval': 2}
    plan, _ = d.freeze_split(
        root,
        config,
        ['data/manifests/previous_use.csv'],
        True,
        'Synthetic prior exposure is recorded for this test fixture.',
    )
    assert exposed_pixel not in set(plan.loc[plan.split.ne('train'), 'pixel_sha256'])


# ---------- features.py ----------

def test_pair_features_shape_order_translation_and_scale_invariance():
    left, right = landmarks(), landmarks(0.5)
    features = f.pair_features([left, right], 640, 480)
    assert len(f.FEATURE_NAMES) == 178
    assert features.shape == (178,)
    assert np.isfinite(features).all()

    np.testing.assert_allclose(features, f.pair_features([right, left], 640, 480))
    shift = np.array([0.1, 0.1, 0.0], dtype=np.float32)
    np.testing.assert_allclose(
        features,
        f.pair_features([left + shift, right + shift], 640, 480),
        atol=1e-5,
    )
    np.testing.assert_allclose(features, f.pair_features([left, right], 1280, 960), atol=1e-5)


@pytest.mark.parametrize('hands', [[], [landmarks()], [np.zeros((21, 3)), landmarks()]])
def test_pair_features_rejects_missing_or_degenerate_hands(hands):
    with pytest.raises(ValueError):
        f.pair_features(hands, 640, 480)


def test_augmentation_is_deterministic_and_non_cropping():
    image = Image.fromarray(np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3))
    first, p1 = f.augment(image, 'a' * 64, 42)
    second, p2 = f.augment(image, 'a' * 64, 42)
    assert p1 == p2
    assert first.tobytes() == second.tobytes()
    assert first.width >= image.width and first.height >= image.height
    assert -6 <= p1['angle'] <= 6
    assert 0.9 <= p1['brightness'] <= 1.1
    assert 0.9 <= p1['contrast'] <= 1.1


def test_samples_augments_train_only(project):
    _, _, plan, _, _ = prepare_split_project(project)
    samples = f.samples(plan, True)
    augmented = samples.loc[samples.variant.ne('original')]
    assert augmented.split.eq('train').all()
    assert augmented.category.isin(['uyir', 'mei']).all()
    assert len(samples.loc[samples.variant.eq('original')]) == len(plan)
    assert not samples.sample_id.duplicated().any()


def single_sample(project):
    root, base = project
    path = add_image(base, '1', 'a.png', 1)
    frame = pd.DataFrame(
        [
            {
                'relative_path': '1/a.png',
                'sha256': sha256_file(path),
                'split': 'train',
                'category': 'ayudha',
            }
        ]
    )
    return root, base, f.samples(frame, False)


@pytest.mark.parametrize('count,status', [(0, 'not_detected'), (1, 'incomplete_pair'), (2, 'ok')])
def test_process_batch_records_detection_status(project, count, status):
    _, base, sample = single_sample(project)
    records, features, lm = f.process_batch(base, sample, FakeDetector([count]), CONFIG)
    assert records[0]['status'] == status
    assert records[0]['n_hands'] == count
    assert lm.shape == (1, 2, 21, 3)
    if count == 2:
        assert np.isfinite(features).all()
    else:
        assert np.isnan(features).all()


def test_retry_is_deterministic_and_recorded(project):
    _, base, sample = single_sample(project)
    detector = FakeDetector([0, 2])
    records, features, _ = f.process_batch(base, sample, detector, dict(CONFIG, retry=True))
    assert detector.calls == 2
    assert records[0]['retry_used'] is True
    assert records[0]['first_n_hands'] == 0
    assert records[0]['n_hands'] == 2
    assert records[0]['status'] == 'ok'
    assert np.isfinite(features).all()


def test_process_batch_rejects_changed_raw_image(project):
    _, base, sample = single_sample(project)
    (base / '1/a.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Rohdaten geaendert'):
        f.process_batch(base, sample, FakeDetector(), CONFIG)


def test_run_extraction_cache_holdout_guard_and_development_bundle(project, monkeypatch):
    root, base, plan, _, _ = prepare_split_project(project)
    model = root / 'models/detector/hand_landmarker_v1.task'
    model.write_bytes(b'fake model')
    monkeypatch.setattr(f, 'model_path', lambda _: model)

    calls = []

    def launch(*args):
        calls.append(1)
        return fake_launch(*args)

    monkeypatch.setattr(f, '_launch', launch)

    development = f.samples(plan.loc[plan.split.ne('test')], True)
    rows, output = f.run_extraction(root, development, CONFIG, 'unit', chunk_size=5)
    assert len(rows) == len(development)
    assert len(calls) > 1
    assert rows.status.eq('ok').all()

    first_call_count = len(calls)
    cached_rows, cached_output = f.run_extraction(root, development, CONFIG, 'unit', chunk_size=5)
    assert len(calls) == first_call_count
    assert cached_output == output
    assert len(cached_rows) == len(rows)

    with pytest.raises(ValueError, match='Holdout'):
        f.run_extraction(root, f.samples(plan, False), CONFIG, 'blocked')

    f.save_feature_decision(
        root,
        CONFIG,
        True,
        'Synthetic feature review confirms deterministic two-hand extraction for tests.',
        True,
    )
    f.save_development_pointer(root, output)
    loaded_rows, matrix, metadata = f.load_development(root)
    assert len(loaded_rows) == len(development)
    assert matrix.shape == (len(development), 178)
    assert metadata['complete'] is True
    assert not loaded_rows.split.eq('test').any()
