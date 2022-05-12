import os
import subprocess
from pathlib import Path


def run(commands):
    # print(" ".join(commands))
    subprocess.run(" ".join(commands), shell=True)


class ExperimentHandler:
    def __init__(self, python_environment, work_directory, run_work_directory, script_location, is_array=False):
        self.script_location = Path(script_location)
        self.run_work_directory = Path(run_work_directory)
        self.work_directory = Path(work_directory)
        self.python_environment = Path(os.getenv("VENV_FOLDER")) / python_environment
        self.run_registry_path = Path(os.getenv("RUN_REGISTRY_LOCATION", "~/run_registry.csv"))
        self.is_array = is_array

        assert ((self.work_directory / self.script_location).exists(),
                f"Script file does not exist. You are trying to run {self.work_directory / self.script_location}.")

        self.modules = ["python/3.8.5"]
        self.run_modules = ["python/3.8.5", "cuda/11.5"]
        self.run_exports = ['export WANDB_MODE="offline"']

        self.additional_lib_installs = [
            ["pip", "install", "torch==1.11.0+cu115", "torchvision==0.12.0+cu115", "-f",
             "https://download.pytorch.org/whl/torch_stable.html"]
        ]

    def load_modules(self):
        run(["module", "purge"])
        for module in self.modules:
            run(["module", "load", module])

    def source_environment(self):
        run(["echo", "Activate environment ${envName}"])
        run(["source", str(self.python_environment / "bin/activate")])

    def setup_experiment(self):
        if (self.work_directory / "setup.py").exists():
            run(["pip", "install", "-e", str(self.work_directory)])
        elif (self.work_directory / "requirements.txt").exists():
            run(["pip", "install", "-r", str(self.work_directory / "requirements.txt")])
        for additional_install in self.additional_lib_installs:
            run(additional_install)
        if (self.work_directory / "offline_setup.py").exists():
            run(["python", str(self.work_directory / "offline_setup")])

    def new_run(self):
        self.source_environment()
        self.load_modules()
        self.setup_experiment()

        # add_run_location = Path(__file__).parent / "register_run.py"
        commands = [
            "jobId=$SLURM_JOB_ID",
            f"runWorkdirJob={str(self.run_work_directory)}/$jobId",
        ]

        if self.is_array:
            commands.append("taskId=$SLURM_ARRAY_TASK_ID")

        commands.extend([
            'mkdir -p "$runWorkdirJob"',
            'mkdir "$runWorkdirJob/logs"',
            'mkdir "$runWorkdirJob/checkpoints"',

            f'cp -r {str(self.work_directory)} $runWorkdirJob',
            f'cd "$runWorkdirJob/{self.work_directory.absolute().name}/{str(self.script_location.parent)}"',
            f'register-run registry={str(self.run_registry_path.absolute())} '
            f'job_id=$jobId status="started" location="$runWorkdirJob"',
            'module purge',
        ])
        commands.extend([f'module load {module}' for module in self.run_modules])
        commands.extend([
            f'source "{str(self.python_environment)}/bin/activate"'
        ])
        commands.extend([export for export in self.run_exports])
        return commands
