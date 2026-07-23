"""Name -> adapter class lookup, so the CLI/configs can select a dataset by
string (e.g. `--dataset synthetic_relational`) without importing every adapter
module directly.
"""

from typing import Dict, Type

from .base import BaseDatasetAdapter
from .ieee_cis import IEEECISAdapter
from .synthetic_relational import SyntheticRelationalAdapter

_REGISTRY: Dict[str, Type[BaseDatasetAdapter]] = {
    "synthetic_relational": SyntheticRelationalAdapter,
    "ieee_cis": IEEECISAdapter,
}


def register_adapter(name: str, adapter_cls: Type[BaseDatasetAdapter]) -> None:
    _REGISTRY[name] = adapter_cls


def get_adapter(name: str) -> BaseDatasetAdapter:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown dataset adapter '{name}'. Available: {available}")
    return _REGISTRY[name]()
