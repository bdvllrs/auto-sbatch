import unittest.mock as mock

from auto_sbatch import auto_sbatch
from tests.utils import mock_for_tests


@mock.patch("auto_sbatch.sbatch.Popen")
def test_start_simple_batch(p_open, capsys):
    mock_for_tests(p_open=p_open)

    slurm_params = {
        "-J": "job-name",
        "-N": 1,
        "--time": "01:00:00"
    }
    run_script = "main.py"

    auto_sbatch(
        "python {script_name} {all_params}",
        slurm_params,
        run_script=run_script,
    )
    captured = capsys.readouterr()
    expected_output = "#!/bin/sh\n"
    for key, val in slurm_params.items():
        if key[1] == "-":  # starts with --
            expected_output += f"#SBATCH {key}={val}\n"
        else:
            expected_output += f"#SBATCH {key} {val}\n"

    expected_output += "\n"
    expected_output += "taskId=0\n"
    expected_output += f"python {run_script} \n"
    expected_output += "Mocked communication output\n"
    expected_output += "Mocked communication error\n"

    assert captured.out == expected_output


@mock.patch("auto_sbatch.sbatch.Popen")
def test_params(p_open, capsys):
    mock_for_tests(p_open=p_open)

    slurm_params = {
        "-J": "job-name",
        "-N": 1,
        "--time": "01:00:00"
    }
    params = {
        "param1": "1",
        "param2": {
            "param2_1": None,
            "param2_2": "a",
        },
        "param3": [1, 2, 3],
    }
    run_script = "main.py"
    command = "python {script_name} {all_params}"

    auto_sbatch(
        command,
        slurm_params,
        params,
        run_script=run_script,
    )

    captured_out = capsys.readouterr().out.strip("\n").split("\n")
    command_out = captured_out[-3]
    expected_command = command.format(
        script_name=run_script,
        all_params='"param1=1" "param2.param2_1=null" '
                   '"param2.param2_2=a" '
                   '"param3=[1, 2, 3]"'
    )
    assert command_out == expected_command


@mock.patch("auto_sbatch.sbatch.Popen")
def test_grid_search_no_array(p_open, capsys):
    mock_for_tests(p_open=p_open)

    slurm_params = {
        "-J": "job-name",
        "-N": 1,
        "--time": "01:00:00"
    }
    params = {
        "param1": [1, 2, 3],
    }
    grid_search = ["param1"]
    run_script = "main.py"
    command = "python {script_name} {all_params}"

    auto_sbatch(
        command,
        slurm_params,
        params,
        grid_search=grid_search,
        run_script=run_script,
    )

    captured_out = capsys.readouterr().out.strip("\n").split("\n")[5:-2]
    for k, key in enumerate(grid_search):
        param_val = '("' + '" "'.join(map(str, params[key])) + '")'
        print(param_val)
        assert captured_out[k] == f"{key}_param={param_val}"
    assert captured_out[-1] == "done"
    assert captured_out[-2].split(" ")[0] == "python"
    assert captured_out[-3] == "do"
    assert captured_out[-4] == "for taskId in $(seq 0 2)"
