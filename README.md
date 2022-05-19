<div align="center">
    <h1>auto-sbatch<br><small>Start SLURM sbatch from config</small></h1>
</div>

`auto-sbatch` creates SLURM scripts automatically and start them using sbatch.

# 🚀 Install

```bash
pip install git+https://github.com/bdvllrs/auto-sbatch.git
```

# Usage

`auto-sbatch` allows to execute sbatch directly from python.
It works by generating a SLURM script on the fly and passing it to sbatch.

## In Python

### `auto_sbatch` function

```python
from auto_sbatch import auto_sbatch

auto_sbatch({
    "-J": "job-name",
    "-N": 1,
    "--time": "01:00:00"
}, {
    "--run-script": "main.py",
})
```

### `SBatch` class

```python
from auto_sbatch import SBatch

sbatch = SBatch({
    "-J": "job-name",
    "-N": 1,
    "--time": "01:00:00",
},{
    "--run-script": "main.py",
    "script-param": 7  # this will be given when the script is run as `python main.py "script-param=7"`
})

sbatch()  # Will add experiment to queue
```

## CLI

```bash
auto-sbatch "slurm.'-J'=job-name" "slurm.'-N'=1" "--run-script=main.py"
```

## Grid-Search
Add a `--grid-search` parameter with a list of the parameter to build the grid-search over.
```bash
auto-sbatch "slurm.'-J'=job-name" "slurm.'-N'=1" "--run-script=main.py" "--grid-search=['param1', 'param2']" "param1=[0, 1]" "param2=[0, 1]"
```

Here, it will start a slurm array of 4 jobs where `param1` and `param2` will have all combinations: (0, 0), (0, 1), (1, 0), (1, 1).

## ExperimentHandler

`auto-sbatch` can do a little more than normal sbatch by setting up an experiment folder.

```python
from pathlib import Path
from auto_sbatch import SBatch, ExperimentHandler

# path to the script to start from the work_directory
# replaces the --run-script
script_location = "main.py"
# The work directory will be copied into an experiment folder. This allows reproducibility
# when looking at the code used for the experiment later.
# Here, we assume the script is at the root of the work directory
work_directory = Path(__file__).absolute().parent
# The work_directory folder will be copied to this folder
run_work_directory = "/path/to/experiments"
# Path to a potential python environment.
python_environment = "/path/to/python/env"
# File where a csv of all run are stored. Not created if not given.
run_registry_path = "~/run_registry.csv"
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
    "-J": "job-name",
    "-N": 1,
    "--time": "01:00:00"
}, {}, handler)

# By default, a script is added to the commands in the SLURM script. You can add other commands that will
# be executed before.
sbatch.add_command("echo $SLURM_JOB_ID")

sbatch()  # batch the experiment!
```

