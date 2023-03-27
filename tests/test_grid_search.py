from omegaconf import OmegaConf

from auto_sbatch.grid_search import _get_grid_combinations

args = OmegaConf.create(
    {
        "a": [1, 2],
        "b": [3, 4],
    }
)


def test_set_grid_combination():
    n_jobs, new_args = _get_grid_combinations(args, ["a", "b"])
    assert n_jobs == 4
    assert new_args.a == [1, 1, 2, 2]
    assert new_args.b == [3, 4, 3, 4]


def test_set_grid_combination_exclude():
    exclude = [{"a": 1, "b": 3}]
    n_jobs, new_args = _get_grid_combinations(args, ["a", "b"], exclude)
    assert n_jobs == 3
    assert new_args.a == [1, 2, 2]
    assert new_args.b == [4, 3, 4]


def test_set_grid_combination_exclude_missing_val():
    exclude = [{"a": 1}]
    n_jobs, new_args = _get_grid_combinations(args, ["a", "b"], exclude)
    assert n_jobs == 2
    assert new_args.a == [2, 2]
    assert new_args.b == [3, 4]
