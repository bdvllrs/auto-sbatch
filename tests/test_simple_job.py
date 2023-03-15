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
