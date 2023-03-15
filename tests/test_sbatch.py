from pathlib import Path

from tests.utils import mock_run

from auto_sbatch import ExperimentHandler, SBatch
import unittest.mock as mock


def mock_for_tests(p_open, subprocess):
    subprocess_instance = mock.MagicMock()
    subprocess_instance.run.side_effect = mock_run
    subprocess.return_value = subprocess_instance
    p_open_instance = mock.MagicMock()
    p_open_instance.communicate.return_value = b"Mocked communication output", b"Mocked communication error"
    p_open.return_value = p_open_instance



@mock.patch("auto_sbatch.processes.subprocess")
@mock.patch("auto_sbatch.sbatch.Popen")
def test_sbatch(p_open, subprocess, capsys):
    mock_for_tests(p_open, subprocess)

    sbatch = SBatch(
        {
            "-J": "job-name",
            "-N": 1,
            "--time": "01:00:00"
        },
        {"script_param": 7},  # this will be given when the script is run as `python main.py "script_param=7"`
        run_script="main.py"
    )

    sbatch("python {script_name} {all_params}")  # Will add experiment to queue


@mock.patch("auto_sbatch.processes.subprocess")
@mock.patch("auto_sbatch.sbatch.Popen")
def test_handled_sbatch(p_open, subprocess, capsys):
    mock_for_tests(p_open, subprocess)

    # path to the script to start from the work_directory
    # replaces the --run-script
    script_location = "test_sbatch.py"
    # The work directory will be copied into an experiment folder. This allows reproducibility
    # when looking at the code used for the experiment later.
    # Here, we assume the script is at the root of the work directory
    work_directory = Path(__file__).absolute().parent
    # The work_directory folder will be copied to this folder
    run_work_directory = "."
    # Path to a potential python environment.
    python_environment = None
    # modules to load BEFORE script are batched
    pre_modules = ["python/3.8.5"]
    # modules to load in the batch system
    run_modules = ["python/3.8.5", "cuda/11.5"]
    # Additional scripts to run before batching the script
    additional_scripts = [
        'pip install numpy'
    ]

    handler = ExperimentHandler(
        script_location,
        work_directory,
        run_work_directory,
        python_environment,
        pre_modules,
        run_modules,
        additional_scripts
    )

    sbatch = SBatch(
        {
            "-J": "job-name",
            "-N": 1,
            "--time": "01:00:00",
        },
        {"script_param": 7},  # this will be given when the script is run as `python main.py "script-param=7"`
        run_script="main.py", experiment_handler=handler
    )

    sbatch("python {script_name} {all_params}")  # Will add experiment to queue
