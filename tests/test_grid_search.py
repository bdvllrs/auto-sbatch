import pytest

from auto_sbatch.grid_search import GridSearch

args = {
    "a": [1, 2],
    "b": [3, 4],
}


def test_set_grid_combination():
    gs = GridSearch(args)
    n_jobs, new_args = gs.get_combinations()
    assert n_jobs == 4
    assert new_args["a"] == [1, 1, 2, 2]
    assert new_args["b"] == [3, 4, 3, 4]


def test_set_grid_combination_one_elem():
    gs = GridSearch({"a": [1, 2]})
    n_jobs, new_args = gs.get_combinations()
    assert n_jobs == 2
    assert new_args["a"] == [1, 2]


def test_set_grid_combination_exclude():
    exclude = [{"a": 1, "b": 3}]
    gs = GridSearch(args, exclude)
    n_jobs, new_args = gs.get_combinations()
    assert n_jobs == 3
    assert new_args["a"] == [1, 2, 2]
    assert new_args["b"] == [4, 3, 4]


def test_set_grid_combination_exclude_missing_val():
    exclude = [{"a": 1}]
    gs = GridSearch(args, exclude)
    n_jobs, new_args = gs.get_combinations()
    assert n_jobs == 2
    assert new_args["a"] == [2, 2]
    assert new_args["b"] == [3, 4]


def test_set_grid_combination_exclude_extra_val():
    exclude = [{"b": [4, 3]}]
    gs = GridSearch({"a": [1, 2]}, exclude)
    with pytest.raises(ValueError):
        gs.get_combinations()
