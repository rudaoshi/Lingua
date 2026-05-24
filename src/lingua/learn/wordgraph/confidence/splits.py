"""Dataset splitting helpers for held-out sample confidence estimation."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class FoldSpec:
    """Paths and metadata for one n-fold confidence split."""

    fold: int
    fold_dir: Path
    data_dir: Path
    train_path: Path
    dev_path: Path
    test_path: Path
    manifest_path: Path
    train_size: int
    dev_size: int
    test_size: int


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into memory."""

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Write rows as JSONL and return the number of rows written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def make_kfold_indices(
    num_samples: int,
    n_folds: int,
    *,
    random_seed: int = 42,
    shuffle: bool = True,
) -> List[List[int]]:
    """Return held-out indices for each fold.

    The implementation intentionally avoids a scikit-learn dependency while
    matching the usual KFold shape: the first folds receive the remainder.
    """

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_folds > num_samples:
        raise ValueError("n_folds cannot exceed the number of samples")

    indices = list(range(num_samples))
    if shuffle:
        rng = random.Random(random_seed)
        rng.shuffle(indices)

    base_size, remainder = divmod(num_samples, n_folds)
    folds: List[List[int]] = []
    start = 0
    for fold_idx in range(n_folds):
        fold_size = base_size + (1 if fold_idx < remainder else 0)
        end = start + fold_size
        folds.append(indices[start:end])
        start = end
    return folds


def _source_path(data_dir: Path, split: str) -> Path:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return path


def _copy_or_write_dev(
    *,
    source_dev_path: Path,
    target_dev_path: Path,
) -> int:
    target_dev_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_dev_path, target_dev_path)
    return sum(1 for line in target_dev_path.open("r", encoding="utf-8") if line.strip())


def prepare_nfold_splits(
    *,
    data_dir: Path | str,
    output_dir: Path | str,
    source_split: str = "train",
    dev_split: str = "dev",
    n_folds: int = 3,
    random_seed: int = 42,
    shuffle: bool = True,
) -> List[FoldSpec]:
    """Create fold data directories for the existing wordgraph trainer.

    Each fold gets a `data/` directory with the `train.jsonl`, `dev.jsonl`, and
    `test.jsonl` files expected by `scripts/train_wordgraph.py`. The held-out
    portion of `source_split` is used as `test.jsonl`; `dev.jsonl` is copied
    from the repo's normal dev split so held-out examples are not used for model
    selection.
    """

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    source_path = _source_path(data_dir, source_split)
    dev_path = _source_path(data_dir, dev_split)

    source_rows = read_jsonl(source_path)
    if not source_rows:
        raise ValueError(f"No samples found in {source_path}")

    heldout_folds = make_kfold_indices(
        len(source_rows),
        n_folds,
        random_seed=random_seed,
        shuffle=shuffle,
    )

    specs: List[FoldSpec] = []
    all_indices = set(range(len(source_rows)))

    for fold_idx, heldout_indices in enumerate(heldout_folds):
        heldout_set = set(heldout_indices)
        train_indices = [i for i in range(len(source_rows)) if i not in heldout_set]
        test_indices = list(heldout_indices)

        fold_dir = output_dir / "folds" / f"fold_{fold_idx}"
        fold_data_dir = fold_dir / "data"
        train_path = fold_data_dir / "train.jsonl"
        dev_out_path = fold_data_dir / "dev.jsonl"
        test_path = fold_data_dir / "test.jsonl"
        manifest_path = fold_data_dir / "manifest.jsonl"

        write_jsonl(train_path, (source_rows[i] for i in train_indices))
        write_jsonl(test_path, (source_rows[i] for i in test_indices))
        dev_size = _copy_or_write_dev(source_dev_path=dev_path, target_dev_path=dev_out_path)

        manifest_rows = []
        for local_idx, global_idx in enumerate(test_indices):
            row = source_rows[global_idx]
            manifest_rows.append(
                {
                    "fold": fold_idx,
                    "fold_local_idx": local_idx,
                    "global_idx": global_idx,
                    "source_split": source_split,
                    "source_line": global_idx + 1,
                    "sentence": row.get("sentence", ""),
                }
            )
        write_jsonl(manifest_path, manifest_rows)

        specs.append(
            FoldSpec(
                fold=fold_idx,
                fold_dir=fold_dir,
                data_dir=fold_data_dir,
                train_path=train_path,
                dev_path=dev_out_path,
                test_path=test_path,
                manifest_path=manifest_path,
                train_size=len(train_indices),
                dev_size=dev_size,
                test_size=len(test_indices),
            )
        )

    covered = set().union(*(set(fold) for fold in heldout_folds))
    if covered != all_indices:
        missing = sorted(all_indices - covered)[:10]
        raise RuntimeError(f"Fold split did not cover all samples; first missing: {missing}")

    return specs


def fold_specs_to_json(specs: Sequence[FoldSpec]) -> List[Dict[str, Any]]:
    """Serialize fold specs for config/summary files."""

    return [
        {
            "fold": spec.fold,
            "fold_dir": str(spec.fold_dir),
            "data_dir": str(spec.data_dir),
            "train_path": str(spec.train_path),
            "dev_path": str(spec.dev_path),
            "test_path": str(spec.test_path),
            "manifest_path": str(spec.manifest_path),
            "train_size": spec.train_size,
            "dev_size": spec.dev_size,
            "test_size": spec.test_size,
        }
        for spec in specs
    ]
