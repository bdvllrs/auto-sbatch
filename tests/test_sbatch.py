import unittest.mock as mock
from pathlib import Path

from auto_sbatch import ExperimentHandler, SBatch
from tests.utils import mock_for_tests


@mock.patch("auto_sbatch.processes.subprocess")
@mock.patch("auto_sbatch.sbatch.Popen")
def test_sbatch(p_open, subprocess, capsys):
    mock_for_tests(p_open=p_open, subprocess=subprocess)

    sbatch = SBatch(
        {"-J": "job-name", "-N": 1, "--time": "01:00:00"},
        {"script_param": 7},
        # this will be given when the script is run as `python main.py
        # "script_param=7"`
        script_name="main.py",
    )

    sbatch.run("python {script_name} {all_params}")  # Will add experiment to
    # queue


@mock.patch("auto_sbatch.processes.subprocess")
@mock.patch("auto_sbatch.sbatch.Popen")
def test_handled_sbatch(p_open, subprocess, capsys):
    mock_for_tests(p_open=p_open, subprocess=subprocess)

    # path to the script to start from the work_directory
    # replaces the --run-script
    script_location = "test_sbatch.py"
    # The work directory will be copied into an experiment folder. This
    # allows reproducibility
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
    additional_scripts = ["pip install numpy"]

    handler = ExperimentHandler(
        script_location,
        work_directory,
        run_work_directory,
        python_environment,
        pre_modules,
        run_modules,
        additional_scripts,
    )

    sbatch = SBatch(
        {
            "-J": "job-name",
            "-N": 1,
            "--time": "01:00:00",
        },
        {"script_param": 7},
        # this will be given when the script is run as `python main.py
        # "script-param=7"`
        script_name="main.py",
        experiment_handler=handler,
    )
    # Will add experiment to queue
    sbatch.run("python {script_name} {all_params}")
