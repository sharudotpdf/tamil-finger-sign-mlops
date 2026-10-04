from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tlfs23 import data as d, visual_review as r
from tlfs23.common import paths, read_csv, read_json, sha256_file, write_json


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    for directory in ['notebooks', 'data/manifests', 'configs', 'artifacts/data_audit']:
        (root / directory).mkdir(parents=True, exist_ok=True)

    base = root / 'data/raw/images'
    mapping = {
        '1': 'அ',
        '13': 'க்',
        '31': 'க',
        '247': 'ஃ',
        'Background': '__background__',
    }

    inventory = []
    for category_index, folder in enumerate(mapping):
        (base / folder).mkdir(parents=True, exist_ok=True)
        for number in range(12):
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            rgb[:, :, 0] = np.arange(32, dtype=np.uint8)[None, :] * 3 + number
            rgb[:, :, 1] = 20 + category_index * 30
            rgb[:, :, 2] = number * 4
            file = base / folder / f'{number:03d}.png'
            Image.fromarray(rgb).save(file)
            inventory.append(
                {
                    'relative_path': file.relative_to(base).as_posix(),
                    'dataset_folder': folder,
                    'filename': file.name,
                    'file_size_bytes': file.stat().st_size,
                    'extension': '.png',
                }
            )

    duplicate = base / '1/copy.png'
    duplicate.write_bytes((base / '1/000.png').read_bytes())
    inventory.append(
        {
            'relative_path': '1/copy.png',
            'dataset_folder': '1',
            'filename': 'copy.png',
            'file_size_bytes': duplicate.stat().st_size,
            'extension': '.png',
        }
    )

    pd.DataFrame(inventory).to_csv(root / 'data/manifests/file_inventory.csv', index=False)
    write_json(
        root / 'artifacts/data_audit/01_inventory_summary.json',
        {'class_root_relative': base.relative_to(root).as_posix()},
    )

    audit, _ = d.audit(root, 'full')
    curated, groups = d.curate(audit)
    d.save_curation(root, curated, groups)
    candidates = d.load_curated(root)

    review = d.label_review(root, candidates)
    review = r.apply_mapping_updates(root, review, mapping)
    r.approve_mapping(root, review, 'Synthetic mapping reference used only for automated tests.')
    review = read_csv(paths(root)['config'] / 'label_review.csv')
    view = r.label_view(review)

    return root, curated, candidates, review, view


CONFIG = {
    'seed': 42,
    'train': 3,
    'focus_train': 3,
    'validation': 2,
    'test': 2,
    'background_eval': 2,
}
NOTE = 'Synthetic review fixture with no previous real-data exposure.'


def freeze(root):
    return r.freeze_or_resume(root, CONFIG, [], True, NOTE, no_previous_exposure=True)


def test_label_view_preserves_mapping_and_components(project):
    _, _, _, review, _ = project
    before = review.copy(deep=True)
    view = r.label_view(review)
    pd.testing.assert_frame_equal(before, review)

    indexed = view.set_index('dataset_folder')
    assert indexed.loc['1', 'vowel_display'] == 'அ'
    assert indexed.loc['1', 'consonant_display'] == 'FIST (Faust)'
    assert indexed.loc['247', 'category'] == 'ayudha'


def test_unknown_label_remains_unknown(project):
    _, _, _, review, _ = project
    changed = review.copy()
    changed.loc[changed.dataset_folder.eq('1'), 'character'] = 'unbekannt'
    view = r.label_view(changed)
    row = view.loc[view.dataset_folder.eq('1')].iloc[0]
    assert pd.isna(row['category'])
    assert row['mapping_status'] == 'NICHT ZUGEORDNET / UNBEKANNT'


def test_mapping_update_resets_verification_and_creates_backup(project):
    root, _, _, review, _ = project
    result = r.apply_mapping_updates(root, review, {'1': 'ஆ'})
    row = result.loc[result.dataset_folder.eq('1')].iloc[0]
    assert not bool(row['verified'])
    assert row['evidence'] == ''
    assert list((paths(root)['work'] / 'reviews').glob('label_review_*.csv'))


def test_folder_examples_supports_paging_and_rejects_unknown_folder(project):
    root, _, candidates, _, view = project
    page = r.folder_examples(root, candidates, view, page=1, page_size=2, per_folder=2)
    assert page.dataset_folder.nunique() == 2
    assert len(page) == 4
    assert r.folder_examples(root, candidates, view, page=100).empty

    with pytest.raises(ValueError, match='Unbekannte Ordner'):
        r.folder_examples(root, candidates, view, folders=['999'])


def test_dhash_pairs_and_notes_are_stable(project):
    root, curated, *_ = project
    synthetic = curated.copy()
    same_folder = synthetic.loc[
        synthetic.dataset_folder.eq('1') & synthetic.use_for_model
    ].head(2).index
    assert len(same_folder) == 2
    synthetic.loc[same_folder, 'dhash'] = '0123456789abcdef'

    pairs = r.dhash_pairs(root, synthetic, max_pairs=3)
    assert not pairs.empty
    assert pairs.hamming_distance.eq(0).all()
    assert not pairs.pixel_identical.any()

    pair_id = pairs.pair_id.iloc[0]
    saved = r.save_dhash_notes(
        root,
        pairs,
        {pair_id: {'assessment': 'UNKLAR', 'note': 'Synthetic review note.'}},
    )
    assert saved.set_index('pair_id').loc[pair_id, 'note'] == 'Synthetic review note.'

    with pytest.raises(ValueError, match='Paar-ID'):
        r.save_dhash_notes(root, pairs, {'does-not-exist': {'assessment': 'UNKLAR'}})


def test_show_cards_records_exposure_and_exports_html(project, monkeypatch):
    import IPython.display

    shown = []
    monkeypatch.setattr(IPython.display, 'display', lambda value: shown.append(value.data))

    root, _, candidates, _, view = project
    preview = r.folder_examples(root, candidates, view, folders=['1'], per_folder=2)
    returned = r.show_cards(root, preview, 'Test', context='unit', output_name='unit_gallery')

    assert len(returned) == 2
    assert shown and 'data:image/png;base64,' in shown[0]
    exposure = read_csv(paths(root)['manifests'] / r.EXPOSURE_NAME)
    assert set(exposure.sha256) == set(preview.sha256)
    assert (paths(root)['work'] / 'reviews/unit_gallery.html').is_file()
    assert (paths(root)['work'] / 'reviews/unit_gallery.csv').is_file()


def test_visual_exposure_is_excluded_from_validation_and_test(project):
    root, _, candidates, _, view = project
    preview = r.folder_examples(root, candidates, view, per_folder=1)
    r._record_exposure(root, preview, 'unit')

    plan, state = freeze(root)
    exposed_pixels = set(preview.pixel_sha256)
    heldout_pixels = set(plan.loc[plan.split.ne('train'), 'pixel_sha256'])
    assert not exposed_pixels & heldout_pixels
    assert any(Path(item['path']).name.startswith('review_exposure_') for item in state['exposure_files'])


def test_resume_preserves_frozen_split_and_changed_config_is_blocked(project):
    root, *_ = project
    plan, state = freeze(root)
    split_file = paths(root)['manifests'] / 'split_manifest.csv'
    before = split_file.read_bytes()

    loaded, state_again = freeze(root)
    assert split_file.read_bytes() == before
    assert state_again['split_sha256'] == state['split_sha256']
    assert loaded.sha256.tolist() == plan.sha256.tolist()

    with pytest.raises(ValueError, match='Konfiguration'):
        r.freeze_or_resume(root, dict(CONFIG, train=4), [], True, NOTE, no_previous_exposure=True)


def test_postsplit_preview_uses_train_only_and_holdout_exposure_is_blocked(project):
    root, _, candidates, _, view = project
    plan, _ = freeze(root)

    preview = r.folder_examples(root, candidates, view)
    assert set(preview.pixel_sha256) <= set(plan.loc[plan.split.eq('train'), 'pixel_sha256'])

    with pytest.raises(ValueError, match='Holdout'):
        r._record_exposure(root, plan.loc[plan.split.eq('test')].head(1), 'unit')


def test_mapping_change_is_blocked_after_split_freeze(project):
    root, _, _, review, _ = project
    freeze(root)
    with pytest.raises(ValueError, match='Mappingaenderung'):
        r.apply_mapping_updates(root, review, {'1': 'ஆ'})


def test_hand_review_requires_all_categories_and_train_only(project):
    root, _, _, _, view = project
    plan, _ = freeze(root)
    examples = r.hand_examples(plan, view)
    assert examples.split.eq('train').all()
    assert set(examples.category) == set(r.CATEGORIES)

    r.approve_hands(
        root,
        examples,
        'Synthetic hand review covers every category and is only for automated testing.',
    )
    hand = read_json(paths(root)['work'] / '04_hand_review.json')
    assert hand['approved'] is True
    assert hand['visual_examples_sha256']

    with pytest.raises(ValueError, match='abdecken'):
        r.approve_hands(
            root,
            examples.loc[examples.category.ne('ayudha')],
            'Synthetic incomplete category review for automated test only.',
        )


def test_show_cards_rejects_modified_image(project):
    root, _, candidates, _, view = project
    preview = r.folder_examples(root, candidates, view, folders=['1'], per_folder=1)
    from tlfs23.common import class_root

    image_path = class_root(root) / preview.relative_path.iloc[0]
    image_path.write_bytes(b'changed')

    with pytest.raises(ValueError, match='Bildpruefung abgebrochen'):
        r.show_cards(root, preview)
