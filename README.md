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

# Main command that will be run
run_command = "python {script_name} {all_params}"
auto_sbatch(run_command, {
    "-J": "job-name",
    "-N": 1,
    "--time": "01:00:00"
}, run_script="main.py")
```

### `SBatch` class

```python
from auto_sbatch import SBatch

sbatch = SBatch({
    "-J": "job-name",
    "-N": 1,
    "--time": "01:00:00",
},{
    "script_param": 7  # this will be given when the script is run as `python main.py "script_param=7"`
}, run_script="main.py")

run_command = "python {script_name} {all_params}"
sbatch(run_command)  # Will add experiment to queue
```

## Grid-Search
Add a `grid_search` parameter with a list of the parameter to build the grid-search over.
```python
from auto_sbatch import SBatch

sbatch = SBatch({
    "-J": "job-name",
    "-N": 1,
    "--time": "01:00:00",
    "--array": "auto"  # this will be automatically changed to the number of generated jobs.
},{
    "param1": [0, 1],
    "param2": [0, 1],
}, grid_search=["param1", "param2"], run_script="main.py")

run_command = "python {script_name} {all_params}"
sbatch(run_command)
```

Here, it will prepare and start 4 jobs where `param1` and `param2` will have all combinations: (0, 0), (0, 1), (1, 0), (1, 1).

A grid search can be executed in several manners:
- if "--array" is provided as a slurm argument, it will be dispatched as a slurm array.
- if "--array" is not provided, it will create only 1 SLURM job and run the jobs sequentially. Be careful to adapt the
length of the job to accommodate all runs. 

If you want to manage the tasks yourself, you can set the `task_id` parameter:

```python
for task_id in range(sbatch.num_available_jobs):
    run_command = "python {script_name} {all_params}"
    task_params = sbatch.get_task_params(task_id)
    sbatch(run_command, task_id)  # will only schedule one task
```

Or you can schedule all tasks one after the other (without a SLURM array):

```python
run_command = "python {script_name} {all_params}"
sbatch(run_command, schedule_all_tasks=True)  # will schedule everything
```

## ExperimentHandler

`auto-sbatch` can do a little more than normal sbatch by setting up an experiment folder.

```python
from pathlib import Path
from auto_sbatch import SBatch, ExperimentHandler

# path to the script to start from the work_directory
# replaces the run_script
script_location = "main.py"
# The work directory will be copied into an experiment folder. This allows reproducibility
# when looking at the code used for the experiment later.
# Here, we assume the script is at the root of the work directory
work_directory = Path(__file__).absolute().parent
# The work_directory folder will be copied to this folder
run_work_directory = "/path/to/experiments"
# Path to a potential python environment.
python_environment = "/path/to/python/env"
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

sbatch = SBatch({
    "-J": "job-name",
    "-N": 1,
    "--time": "01:00:00"
}, experiment_handler=handler)

# By default, a script is added to the commands in the SLURM script. You can add other commands that will
# be executed before.
sbatch.add_command("echo $SLURM_JOB_ID")

sbatch("python {script_name} {all_params}")  # batch the experiment!
```

### Available shortcuts for `run_command`
- `{script_name}` path to script
- `{params}` provided params, excluding `{grid_search_params}`
- `{grid_search_params}` parameters computed by grid_search as parameter format
- `{grid_search_string}` parameters computed by grid_search in a string form
- `{all_params}` combines `{params}` and `{grid_search_params}`
- `{num_gpus}` number of requested gpus to slurm.

When using the experiment handler:
- `{checkpoints_dir}` location to the checkpoint directory.
