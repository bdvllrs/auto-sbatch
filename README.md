<div align="center">
    <h1>auto-sbatch<br><small>Start SLURM sbatch from config</small></h1>
</div>

`auto-sbatch` creates SLURM scripts automatically and start them using sbatch.

# 🚀 Install
```
pip install git+https://github.com/bdvllrs/auto-sbatch.git
```

# Usage
```
auto-sbatch "-J=job-name" "-N=1" "python_environment=my-env" "work_directory=/path/to/my/work/dir" "run_work_directory=/path/to/saved/experiments" "script_location=relative/path/start/script"
```


