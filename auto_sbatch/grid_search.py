from itertools import product

from omegaconf import ListConfig, OmegaConf

from auto_sbatch.utils import get_dotlist_params


class GridSearch:
    def __init__(self, selected_params, exclude=None):
        self._selected_params = selected_params
        self._exclude = exclude or []

    def get_combinations(self, params):
        n_jobs, params = _get_grid_combinations(
            params, self._selected_params, self._exclude
        )
        return n_jobs, params

    def __contains__(self, item):
        return item in self._selected_params


def _is_excluded(keys, values, excluded):
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


def _get_grid_combinations(args, grid_search, exclude=None):
    def cond(key, val):
        return key in grid_search and isinstance(val, ListConfig)

    unstructured_dict = get_dotlist_params(args, cond=cond)

    keys, values = zip(*unstructured_dict.items())
    all_combinations = list(product(*values))
    if exclude is not None:
        all_combinations = [
            comb for comb in all_combinations
            if not _is_excluded(keys, comb, exclude)
        ]
    n_jobs = len(all_combinations)
    new_dict = []
    for k, key in enumerate(keys):
        combination = ', '.join(
            [str(all_combinations[i][k])
             for i in range(len(all_combinations))]
        )
        new_dict.append(
            f"{key}=[{combination}]"
        )
    new_values = OmegaConf.from_dotlist(new_dict)
    return n_jobs, OmegaConf.merge(args, new_values)
