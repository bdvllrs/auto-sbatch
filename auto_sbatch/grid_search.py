from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any


class GridSearch:
    def __init__(
        self,
        values: Mapping[str, Sequence[Any]],
        exclude: Sequence[Mapping[str, Any]] | None = None,
    ):
        self._values = values
        self._exclude = exclude or []
        self.n_jobs, self.combinations = self.get_combinations()

    def get_combinations(self) -> tuple[int, dict[str, list[Any]]]:
        keys, values = tuple(zip(*self._values.items()))
        all_combinations = list(product(*values))
        all_combinations = [
            comb
            for comb in all_combinations
            if not _is_excluded(keys, comb, self._exclude)
        ]
        njobs = len(all_combinations)
        new_values = {}
        for k, key in enumerate(keys):
            new_values[key] = [comb[k] for comb in all_combinations]
        return njobs, new_values

    def job_params(self, job_id: int) -> dict[str, Any]:
        if 0 < job_id or job_id >= self.n_jobs:
            raise ValueError(f"job_id should be >= 0 and < {self.n_jobs}")
        return {key: val[job_id] for key, val in self.combinations.items()}

    def __contains__(self, item: str) -> bool:
        return item in self._values


def _is_excluded(
    keys: Sequence[str], values: Sequence[Any], excluded: Sequence[Mapping[str, Any]]
) -> bool:
    for excluded_item in excluded:
        if not set(excluded_item.keys()).issubset(set(keys)):
            raise ValueError(
                "Keys of excluded item must be a subset of keys used "
                "for grid-search."
            )
        for key, value in zip(keys, values):
            if key in excluded_item.keys() and value != excluded_item[key]:
                break
        else:
            return True
    return False
