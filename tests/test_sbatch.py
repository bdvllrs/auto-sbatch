from pathlib import Path
from .utils import mock_processes
from auto_sbatch import SBatch, ExperimentHandler


def test_sbatch(mock_processes, capsys):
    sbatch = SBatch({
        "--run-script": "main.py",
        "-J": "job-name",
        "-N": 1,
        "--time": "01:00:00",
        "script-param": 7  # this will be given when the script is run as `python main.py "script-param=7"`
    })

    sbatch()  # Will add experiment to queue


def test_handled_sbatch(mock_processes, capsys):
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
    # File where a csv of all run are stored. Not created if not given.
    run_registry_path = None
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
        run_registry_path,
        pre_modules,
        run_modules,
        additional_scripts
    )

    sbatch = SBatch({
        "--run-script": "main.py",
        "-J": "job-name",
        "-N": 1,
        "--time": "01:00:00",
        "script-param": 7  # this will be given when the script is run as `python main.py "script-param=7"`
    }, handler)

    sbatch()  # Will add experiment to queue
