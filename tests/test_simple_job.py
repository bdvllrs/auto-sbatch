from auto_sbatch import auto_sbatch
from auto_sbatch.sbatch import main
from .utils import mock_processes


def test_start_simple_batch(mock_processes, capsys):
    auto_sbatch("python {script_name} {all_params}", {
        "run_script": "main.py",
        "slurm": {
            "-J": "job-name",
            "-N": 1,
            "--time": "01:00:00"
        }
    })
    captured = capsys.readouterr()
    assert captured.out == """Running generated SLURM script:
#!/bin/sh
#SBATCH -J job-name
#SBATCH -N 1
#SBATCH --time=01:00:00

taskId=0
python "main.py" "gpus=0" "--run-script=main.py" "checkpoints_dir='../../checkpoints/$jobId'"
Mocked communication output
Mocked communication error
"""


def test_cli():
    main()