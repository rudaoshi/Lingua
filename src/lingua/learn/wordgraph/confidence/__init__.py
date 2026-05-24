"""N-fold confidence utilities for the Lingua wordgraph parser."""

from .merge import merge_fold_results
from .splits import FoldSpec, prepare_nfold_splits

__all__ = ["FoldSpec", "merge_fold_results", "prepare_nfold_splits"]
