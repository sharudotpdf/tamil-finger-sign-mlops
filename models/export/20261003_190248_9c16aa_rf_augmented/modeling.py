"""Modelltraining, Validation, Threshold-Auswahl und MLflow fuer TLFS23.

Dieses Modul setzt den freigegebenen Datenstand aus ``tlfs23.features`` voraus.
Es erzeugt keine Splits und extrahiert keine Bilder erneut. Dadurch bleibt die
Verantwortung klar getrennt:

- data.py: Audit, Labels und Split
- features.py: MediaPipe und 178 geometrische Merkmale
- modeling.py: Training, Validation, Modellauswahl und Tracking
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from . import data as d
from . import features as f
from .common import paths, read_json, runtime, sha256_file, write_json

BG = d.BG
NA = d.NA
FIST = d.FIST
REJECT = "__reject__"
COMPONENT_TYPES = ("uyir", "mei", "uyirmei")


def _schema_maps() -> tuple[dict[tuple[str, str, str], str], str]:
    """Erzeugt die erlaubte strukturierte Zuordnung direkt aus dem kanonischen Schema."""
    schema = d.canonical_schema().copy()
    structured = schema.loc[schema["category"].isin(COMPONENT_TYPES)]
    mapping = {
        (str(row.category), str(row.vowel_target), str(row.consonant_target)): str(row.character)
        for row in structured.itertuples()
    }
    ayudha = str(schema.loc[schema["category"].eq("ayudha"), "character"].iloc[0])
    return mapping, ayudha


COMPONENT_TO_CHARACTER, AYUDHA = _schema_maps()


def _model_runtime() -> dict:
    info = runtime()
    packages = dict(info.get("packages", {}))
    for name in ["mlflow", "joblib", "threadpoolctl"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    info["packages"] = packages
    return info


def _git_state(root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
            ).strip()
        )
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not_committed", True


def load_development(root: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Laedt ausschliesslich den in Notebook 05 freigegebenen Entwicklungsstand."""
    frame, x, metadata = f.load_development(root)

    required = {
        "relative_path",
        "sha256",
        "split",
        "variant",
        "character",
        "category",
        "vowel_target",
        "consonant_target",
        "status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Entwicklungsdaten ohne erforderliche Spalten: {sorted(missing)}")

    if "test" in set(frame["split"]):
        raise ValueError("Notebook 06 darf keine Testdaten enthalten.")
    if x.shape != (len(frame), len(f.FEATURE_NAMES)):
        raise ValueError(
            f"Feature-Matrix hat {x.shape}; erwartet {(len(frame), len(f.FEATURE_NAMES))}."
        )
    if set(frame["variant"]) - {"original", "mild_1"}:
        raise ValueError("Unbekannte Trainingsvariante im Entwicklungsartefakt.")
    if frame.loc[frame["split"].eq("validation"), "variant"].ne("original").any():
        raise ValueError("Validation muss unveraendert bleiben; Augmentation nur im Training.")

    return frame, x, metadata


def _training_view(
    frame: pd.DataFrame,
    x: np.ndarray,
    include_augmentation: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = frame["split"].eq("train")
    if not include_augmentation:
        mask &= frame["variant"].eq("original")
    train = frame.loc[mask].reset_index(drop=True)
    x_train = x[mask.to_numpy()]
    if train.empty:
        raise ValueError("Leere Trainingsauswahl.")
    return train, x_train


def fit_heads(
    frame: pd.DataFrame,
    x: np.ndarray,
    family: str,
    seed: int,
    rf_trees: int = 120,
) -> dict:
    """Trainiert Kategorie-, Vokal- und Konsonantenkopf auf gueltigen Features.

    Background mit zwei erkannten Haenden wird bewusst in den Kategorie-Kopf
    aufgenommen. Die Komponentenkoepfe lernen nur Uyir/Mei/Uyirmei.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if x.shape != (len(frame), len(f.FEATURE_NAMES)):
        raise ValueError("Training und Feature-Matrix sind nicht ausgerichtet.")

    heads: dict[str, object] = {}
    target_masks: dict[str, np.ndarray] = {}

    for target in ["category", "vowel_target", "consonant_target"]:
        mask = frame["status"].eq("ok")
        if target != "category":
            mask &= frame["category"].isin(COMPONENT_TYPES)

        y = frame.loc[mask, target].astype(str).to_numpy()
        x_target = x[mask.to_numpy()]

        if len(y) == 0:
            raise ValueError(f"Keine Trainingsbeispiele fuer {target}.")
        if not np.isfinite(x_target).all():
            raise ValueError(f"Nicht-finite Features im Training fuer {target}.")

        if family == "dummy":
            estimator = DummyClassifier(strategy="most_frequent")
        elif family == "logistic_regression":
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
        elif family == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=rf_trees,
                max_depth=18,
                min_samples_leaf=2,
                max_features="sqrt",
                max_samples=0.8,
                class_weight="balanced_subsample",
                n_jobs=2,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unbekannte Modellfamilie: {family}")

        estimator.fit(x_target, y)
        heads[target] = estimator
        target_masks[target] = mask.to_numpy()

    sign_characters = sorted(set(frame.loc[frame["character"].ne(BG), "character"].astype(str)))

    return {
        "heads": heads,
        "family": family,
        "supported_characters": sign_characters,
        "feature_names": list(f.FEATURE_NAMES),
        "n_features": len(f.FEATURE_NAMES),
        "n_training_rows": int(len(frame)),
        "n_valid_feature_rows": int(frame["status"].eq("ok").sum()),
    }


def _head_predictions(estimator, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = estimator.predict_proba(x)
    choices = probabilities.argmax(axis=1)
    return estimator.classes_[choices], probabilities[np.arange(len(choices)), choices]


def predict_heads(
    bundle: dict,
    frame: pd.DataFrame,
    x: np.ndarray,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Erzeugt strukturierte Vorhersagen ohne Ground-Truth fuer Inferenzentscheidungen."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold muss zwischen 0 und 1 liegen.")
    if x.shape != (len(frame), len(f.FEATURE_NAMES)):
        raise ValueError("Feature-Matrix passt nicht zu den Vorhersagezeilen.")

    keep = [
        "relative_path",
        "sha256",
        "character",
        "category",
        "vowel_target",
        "consonant_target",
        "status",
        "variant",
        "split",
    ]
    optional = [c for c in ["dataset_folder", "n_hands"] if c in frame.columns]
    result = frame[keep + optional].reset_index(drop=True).copy()

    result["pred_type"] = REJECT
    result["pred_vowel"] = REJECT
    result["pred_consonant"] = REJECT
    result["raw_character"] = REJECT
    result["score"] = 0.0
    result["reason"] = result["status"].astype(str)

    valid_indices = np.flatnonzero(frame["status"].eq("ok").to_numpy())
    if len(valid_indices):
        x_valid = x[valid_indices]
        estimates = {
            target: _head_predictions(estimator, x_valid)
            for target, estimator in bundle["heads"].items()
        }

        supported = set(bundle["supported_characters"])

        for local_index, row_index in enumerate(valid_indices):
            category = str(estimates["category"][0][local_index])
            category_score = float(estimates["category"][1][local_index])
            vowel = str(estimates["vowel_target"][0][local_index])
            vowel_score = float(estimates["vowel_target"][1][local_index])
            consonant = str(estimates["consonant_target"][0][local_index])
            consonant_score = float(estimates["consonant_target"][1][local_index])

            result.loc[row_index, ["pred_type", "pred_vowel", "pred_consonant"]] = [
                category,
                vowel,
                consonant,
            ]

            if category == "background":
                char = BG
                score = category_score
                reason = "background_category"
                result.loc[row_index, ["pred_vowel", "pred_consonant"]] = [NA, NA]
            elif category == "ayudha":
                char = AYUDHA
                score = category_score
                reason = "ayudha_category"
                result.loc[row_index, ["pred_vowel", "pred_consonant"]] = [NA, NA]
            elif category in COMPONENT_TYPES:
                char = COMPONENT_TO_CHARACTER.get((category, vowel, consonant), REJECT)
                score = min(category_score, vowel_score, consonant_score)
                reason = "component_composition" if char != REJECT else "inconsistent_components"
            else:
                char = REJECT
                score = category_score
                reason = "unknown_category"

            if char not in {REJECT, BG} and char not in supported:
                char = REJECT
                reason = "outside_supported_scope"

            result.loc[row_index, ["raw_character", "score", "reason"]] = [
                char,
                score,
                reason,
            ]

    return apply_threshold(result, threshold)


def apply_threshold(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = predictions.copy()
    result["accepted"] = result["raw_character"].ne(REJECT) & result["score"].ge(threshold)
    result["predicted_character"] = result["raw_character"].where(result["accepted"], REJECT)
    below = result["raw_character"].ne(REJECT) & ~result["accepted"]
    result.loc[below, "reason"] = "below_score_threshold"
    return result


def evaluate_predictions(predictions: pd.DataFrame, sign_labels: list[str]) -> dict:
    """Bewertet Klassifikator und Gesamtsystem getrennt.

    Detektionsausfaelle bei echten Zeichen bleiben End-to-End-Fehler. Background
    darf Haende enthalten: als Fehlannahme gilt nur ein akzeptiertes Tamil-Zeichen.
    """
    from sklearn.metrics import f1_score

    p = predictions
    signs = p["character"].ne(BG)
    background = ~signs
    accepted = p["accepted"].astype(bool)
    correct = p["predicted_character"].eq(p["character"])
    detected = p["status"].eq("ok")
    accepted_sign_inputs = signs & accepted

    def ratio(numerator: int | np.integer, denominator: int | np.integer):
        return float(numerator / denominator) if denominator else None

    false_sign_on_background = (
        background
        & accepted
        & p["predicted_character"].ne(BG)
        & p["predicted_character"].ne(REJECT)
    )

    metrics = {
        "n_inputs": int(len(p)),
        "n_sign_inputs": int(signs.sum()),
        "n_background_inputs": int(background.sum()),
        "e2e_macro_f1": float(
            f1_score(
                p.loc[signs, "character"],
                p.loc[signs, "predicted_character"],
                labels=sign_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "e2e_sign_accuracy": ratio((correct & signs).sum(), signs.sum()),
        "sign_coverage": ratio(accepted_sign_inputs.sum(), signs.sum()),
        "accepted_sign_accuracy": ratio(
            (correct & accepted_sign_inputs).sum(),
            accepted_sign_inputs.sum(),
        ),
        "n_accepted_signs": int(accepted_sign_inputs.sum()),
        "background_false_accept_rate": ratio(
            false_sign_on_background.sum(),
            background.sum(),
        ),
        "background_explicit_no_sign_rate": ratio(
            (background & p["predicted_character"].eq(BG)).sum(),
            background.sum(),
        ),
        "background_safe_reject_or_no_sign_rate": ratio(
            (
                background
                & p["predicted_character"].isin([BG, REJECT])
            ).sum(),
            background.sum(),
        ),
        "two_hand_rate_signs": ratio((signs & detected).sum(), signs.sum()),
        "two_hand_rate_background": ratio((background & detected).sum(), background.sum()),
        "n_detection_failures_signs": int((signs & ~detected).sum()),
        "reject_rate": float((~accepted).mean()),
    }

    detected_signs = signs & detected
    if detected_signs.any():
        metrics["detected_sign_macro_f1"] = float(
            f1_score(
                p.loc[detected_signs, "character"],
                p.loc[detected_signs, "predicted_character"],
                labels=sign_labels,
                average="macro",
                zero_division=0,
            )
        )
        metrics["detected_sign_accuracy"] = ratio(
            (correct & detected_signs).sum(),
            detected_signs.sum(),
        )
    else:
        metrics["detected_sign_macro_f1"] = None
        metrics["detected_sign_accuracy"] = None

    return metrics


def choose_threshold(
    raw_predictions: pd.DataFrame,
    labels: list[str],
    target_accepted_sign_accuracy: float = 0.95,
    max_background_false_accept_rate: float = 0.05,
    min_accepted_signs: int = 30,
) -> tuple[float, pd.DataFrame, bool]:
    """Waehlt den Operating Point ausschliesslich auf Validation."""
    grid = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    rows = []

    for threshold in grid:
        metrics = evaluate_predictions(apply_threshold(raw_predictions, threshold), labels)
        rows.append({"threshold": threshold, **metrics})

    table = pd.DataFrame(rows)
    eligible = table[
        (table["accepted_sign_accuracy"] >= target_accepted_sign_accuracy)
        & (table["n_accepted_signs"] >= min_accepted_signs)
        & (table["background_false_accept_rate"] <= max_background_false_accept_rate)
    ]

    target_met = not eligible.empty
    if target_met:
        chosen = eligible.sort_values(
            ["sign_coverage", "e2e_macro_f1", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
    else:
        chosen = table.sort_values(
            ["e2e_macro_f1", "sign_coverage", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]

    return float(chosen["threshold"]), table, target_met


def default_experiments() -> list[dict]:
    """Kleine, vorab festgelegte Experimentmenge ohne breite Hyperparametersuche."""
    return [
        {"name": "dummy_original", "family": "dummy", "include_augmentation": False},
        {"name": "logreg_original", "family": "logistic_regression", "include_augmentation": False},
        {"name": "logreg_augmented", "family": "logistic_regression", "include_augmentation": True},
        {"name": "rf_original", "family": "random_forest", "include_augmentation": False},
        {"name": "rf_augmented", "family": "random_forest", "include_augmentation": True},
    ]


def train_experiments(
    root: Path,
    seed: int = 42,
    target_accepted_sign_accuracy: float = 0.95,
    max_background_false_accept_rate: float = 0.05,
    min_accepted_signs: int = 30,
    track: bool = True,
    rf_trees: int = 120,
    experiments: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Trainiert nur auf Train und waehlt Modell/Threshold nur auf Validation."""
    import joblib
    from threadpoolctl import threadpool_limits

    root = Path(root).resolve()
    p = paths(root)
    model_root = p["work"] / "modeling"
    model_root.mkdir(parents=True, exist_ok=True)

    if (p["work"] / "07_test_access_lock.json").exists():
        raise RuntimeError("Finaler Test wurde bereits geoeffnet; Notebook 06 nicht weiter tunen.")

    frame, x, feature_metadata = load_development(root)
    validation_mask = frame["split"].eq("validation").to_numpy()
    validation = frame.loc[validation_mask].reset_index(drop=True)
    x_validation = x[validation_mask]

    if validation.empty or validation["variant"].ne("original").any():
        raise ValueError("Validation fehlt oder enthaelt augmentierte Varianten.")

    sign_labels = sorted(set(frame.loc[frame["character"].ne(BG), "character"].astype(str)))
    observed_train = set(
        frame.loc[
            frame["split"].eq("train") & frame["status"].eq("ok") & frame["character"].ne(BG),
            "character",
        ].astype(str)
    )
    missing_train = set(sign_labels) - observed_train
    if missing_train:
        raise ValueError(
            "Fuer einige Zeichen existiert keine vollstaendige Handdetektion im Training: "
            + str(sorted(missing_train)[:20])
        )

    specs = experiments or default_experiments()
    names = [spec["name"] for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Experimentnamen muessen eindeutig sein.")

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    run_dir = model_root / "experiments" / batch_id
    run_dir.mkdir(parents=True, exist_ok=False)

    git_commit, git_dirty = _git_state(root)
    split_plan, split_state = d.load_split(root)
    del split_plan
    feature_decision_path = p["work"] / "05_feature_decision.json"
    feature_pointer_path = p["work"] / "development_features.json"

    provenance = {
        "seed": seed,
        "rf_trees": rf_trees,
        "split_sha256": split_state["split_sha256"],
        "feature_decision_sha256": sha256_file(feature_decision_path),
        "development_pointer_sha256": sha256_file(feature_pointer_path),
        "feature_metadata_sha256": sha256_file(
            root / read_json(feature_pointer_path)["directory"] / "metadata.json"
        ),
        "features_code_sha256": sha256_file(Path(f.__file__)),
        "modeling_code_sha256": sha256_file(Path(__file__)),
        "runtime": _model_runtime(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    write_json(run_dir / "experiment_provenance.json", provenance)

    mlflow = None
    if track:
        import mlflow as mf

        mlflow = mf
        tracking_uri = "sqlite:///" + (model_root / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = "TLFS23_structured_178_features"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                experiment_name,
                artifact_location=(model_root / "mlflow_artifacts").as_uri(),
            )
        mlflow.set_experiment(experiment_name)

    records = []

    for spec in specs:
        experiment_name = str(spec["name"])
        family = str(spec["family"])
        include_augmentation = bool(spec["include_augmentation"])

        train, x_train = _training_view(frame, x, include_augmentation)
        training_data = "original+mild_1" if include_augmentation else "original"

        subdir = run_dir / experiment_name
        subdir.mkdir()

        start = time.perf_counter()
        with threadpool_limits(limits=2):
            bundle = fit_heads(train, x_train, family, seed, rf_trees)
            fit_seconds = time.perf_counter() - start

            # Warm-up wird nicht in die Batch-Laufzeit aufgenommen.
            _ = predict_heads(bundle, validation.iloc[:1], x_validation[:1])
            prediction_start = time.perf_counter()
            raw_predictions = predict_heads(bundle, validation, x_validation, threshold=0.0)
            classifier_ms_per_image = (
                1000 * (time.perf_counter() - prediction_start) / len(validation)
            )

        metrics = evaluate_predictions(raw_predictions, sign_labels)
        bundle.update(
            {
                "experiment_name": experiment_name,
                "training_data": training_data,
                "include_augmentation": include_augmentation,
                "seed": seed,
                "rf_trees": rf_trees if family == "random_forest" else 0,
                "feature_metadata": feature_metadata,
                "provenance": provenance,
                "model_version": f"{batch_id}_{experiment_name}",
            }
        )

        model_file = subdir / "model.joblib"
        joblib.dump(bundle, model_file, compress=3)
        raw_predictions.to_csv(subdir / "validation_predictions_threshold0.csv", index=False)

        metrics.update(
            {
                "fit_seconds": float(fit_seconds),
                "classifier_batch_ms_per_image": float(classifier_ms_per_image),
                "model_mib": float(model_file.stat().st_size / 1024**2),
                "n_train_rows": int(len(train)),
                "n_train_ok": int(train["status"].eq("ok").sum()),
            }
        )
        write_json(subdir / "metrics_threshold0.json", metrics)

        run_id = "tracking-disabled"
        if mlflow is not None:
            with mlflow.start_run(run_name=bundle["model_version"]) as run:
                run_id = run.info.run_id
                mlflow.log_params(
                    {
                        "experiment_name": experiment_name,
                        "family": family,
                        "training_data": training_data,
                        "include_augmentation": include_augmentation,
                        "seed": seed,
                        "rf_trees": rf_trees if family == "random_forest" else 0,
                        "n_features": len(f.FEATURE_NAMES),
                        "n_train_rows": len(train),
                        "n_validation_rows": len(validation),
                        "test_accessed": False,
                        "split_sha256": provenance["split_sha256"],
                        "feature_decision_sha256": provenance["feature_decision_sha256"],
                    }
                )
                mlflow.set_tags(
                    {
                        "git_commit": git_commit,
                        "git_dirty": str(git_dirty),
                        "scope": "train_validation_only",
                        "modeling_code_sha256": provenance["modeling_code_sha256"],
                        "features_code_sha256": provenance["features_code_sha256"],
                    }
                )
                mlflow.log_metrics(
                    {
                        key: float(value)
                        for key, value in metrics.items()
                        if value is not None
                        and isinstance(value, (int, float, np.integer, np.floating))
                        and np.isfinite(value)
                    }
                )
                mlflow.log_artifacts(str(subdir))

        records.append(
            {
                "experiment": experiment_name,
                "family": family,
                "training_data": training_data,
                "include_augmentation": include_augmentation,
                "mlflow_run_id": run_id,
                "model_path": str(model_file.relative_to(root)),
                **metrics,
            }
        )

    results = (
        pd.DataFrame(records)
        .sort_values(
            ["e2e_macro_f1", "model_mib", "experiment"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    results.to_csv(run_dir / "validation_comparison.csv", index=False)

    winner = results.iloc[0]
    winner_model_path = root / str(winner["model_path"])
    bundle = joblib.load(winner_model_path)
    raw_predictions = predict_heads(bundle, validation, x_validation, threshold=0.0)

    threshold, threshold_table, target_met = choose_threshold(
        raw_predictions,
        sign_labels,
        target_accepted_sign_accuracy=target_accepted_sign_accuracy,
        max_background_false_accept_rate=max_background_false_accept_rate,
        min_accepted_signs=min_accepted_signs,
    )
    threshold_path = run_dir / "threshold_validation.csv"
    threshold_table.to_csv(threshold_path, index=False)

    final_validation = evaluate_predictions(
        apply_threshold(raw_predictions, threshold),
        sign_labels,
    )
    write_json(run_dir / "selected_validation_metrics.json", final_validation)

    selection = {
        "experiment": str(winner["experiment"]),
        "family": str(winner["family"]),
        "training_data": str(winner["training_data"]),
        "model_path": str(winner["model_path"]),
        "model_sha256": sha256_file(winner_model_path),
        "model_version": bundle["model_version"],
        "mlflow_run_id": str(winner["mlflow_run_id"]),
        "threshold": threshold,
        "target_accepted_sign_accuracy": target_accepted_sign_accuracy,
        "max_background_false_accept_rate": max_background_false_accept_rate,
        "min_accepted_signs": min_accepted_signs,
        "validation_target_met": target_met,
        "chosen_on": "validation e2e_macro_f1; tie: smaller model artifact",
        "threshold_chosen_on": "validation only",
        "source_experiment_dir": str(run_dir.relative_to(root)),
        "threshold_table_path": str(threshold_path.relative_to(root)),
        "validation_metrics": final_validation,
        "provenance": provenance,
        "test_accessed": False,
    }

    selection_path = model_root / "selected_model.json"
    write_json(selection_path, selection)

    if mlflow is not None and selection["mlflow_run_id"] != "tracking-disabled":
        with mlflow.start_run(run_id=selection["mlflow_run_id"]):
            mlflow.log_params(
                {
                    "selected_threshold": threshold,
                    "validation_target_met": target_met,
                }
            )
            mlflow.log_artifact(str(threshold_path))
            mlflow.log_artifact(str(run_dir / "selected_validation_metrics.json"))

    print(
        "Ausgewaehlt:",
        selection["experiment"],
        "| Validation-Ziel erreicht:",
        target_met,
    )
    return results, selection


def save_model_decision(root: Path, note: str) -> dict:
    """Friert die begruendete Modellentscheidung vor dem Holdout-Test ein."""
    if len(note.strip()) < 30 or "..." in note:
        raise ValueError("Eine konkrete Modellentscheidung mit mindestens 30 Zeichen dokumentieren.")

    root = Path(root).resolve()
    p = paths(root)
    selection_path = p["work"] / "modeling" / "selected_model.json"
    selection = read_json(selection_path)

    decision = {
        "model_version": selection["model_version"],
        "experiment": selection["experiment"],
        "family": selection["family"],
        "training_data": selection["training_data"],
        "threshold": selection["threshold"],
        "validation_target_met": selection["validation_target_met"],
        "note": note.strip(),
        "test_used_for_selection": False,
        "selection_sha256": sha256_file(selection_path),
    }

    target = p["work"] / "06_model_decision.json"
    if target.exists() and read_json(target) != decision:
        raise ValueError(
            "Modellentscheidung wurde bereits festgeschrieben. Alten Stand zuerst archivieren."
        )
    write_json(target, decision)
    return decision


def mlflow_ui_command(root: Path, port: int = 5001) -> str:
    root = Path(root).resolve()
    database = paths(root)["work"] / "modeling" / "mlflow.db"
    relative = database.relative_to(root).as_posix()
    return f"mlflow ui --backend-store-uri sqlite:///{relative} --port {port}"
