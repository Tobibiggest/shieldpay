"""Base contract every dataset adapter implements.

An adapter's job is narrow: read a dataset's native files and produce (a) a
dataframe and (b) a `FraudDatasetSchema` describing it. Everything downstream
(graph construction, feature preprocessing, training, evaluation) depends only
on the schema, never on the adapter or the dataset's original column names --
that's what lets a new dataset plug in via a new adapter alone.
"""

from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Union

import pandas as pd

from ...schema import FraudDatasetSchema


class BaseDatasetAdapter(ABC):
    @abstractmethod
    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        """Read the dataset's native file(s) into a single dataframe."""

    @abstractmethod
    def get_schema(self) -> FraudDatasetSchema:
        """Describe which columns of `load()`'s output play which role, using
        this dataset's own (pre-rename) column names -- pair with `load()`,
        not `load_canonical()` (see `get_canonical_schema`)."""

    def to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename this dataset's ID/label/timestamp columns to the canonical
        names (`sender_id`, `device_id`, `label`, ...) that the graph builder
        and training code target. Feature columns are left as named in the
        schema -- only relational/label/timestamp columns are canonicalized.
        """
        schema = self.get_schema()
        rename_map = {}
        for attr, canonical in FraudDatasetSchema.CANONICAL_NAMES.items():
            source_col = getattr(schema, attr, None)
            if source_col is not None and source_col in df.columns and source_col != canonical:
                rename_map[source_col] = canonical
        return df.rename(columns=rename_map)

    def get_canonical_schema(self) -> FraudDatasetSchema:
        """The schema to use alongside `load_canonical()`'s output: every
        ID/label/timestamp field renamed to its canonical name, matching what
        `to_canonical()` actually did to the dataframe. Using `get_schema()`
        (this dataset's native names) together with `load_canonical()`'s
        renamed dataframe is a mismatch that silently breaks for any adapter
        whose native names differ from the canonical ones -- always pair
        `load()` with `get_schema()`, or `load_canonical()` with this method.
        """
        schema = self.get_schema()
        updates = {
            attr: canonical
            for attr, canonical in FraudDatasetSchema.CANONICAL_NAMES.items()
            if getattr(schema, attr, None) is not None
        }
        return replace(schema, **updates)

    def load_canonical(self, path: Union[str, Path]) -> pd.DataFrame:
        return self.to_canonical(self.load(path))
