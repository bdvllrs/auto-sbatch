from pathlib import Path

from auto_sbatch.processes import run


class ExperimentHandler:
    def __init__(self,
                 script_location,
                 work_directory=".",
                 run_work_directory=".",
                 python_environment=None,
                 pre_modules=None,
                 run_modules=None,
                 additional_scripts=None,
                 setup_experiment=True,
                 use_conda=False):
        self.script_location = Path(script_location)
        self.run_work_directory = Path(run_work_directory)
        self.work_directory = Path(work_directory)
        self.python_environment = python_environment

        self._setup_experiment = setup_experiment
        self._use_conda = use_conda

        if not (self.work_directory / self.script_location).exists():
            raise ValueError(f"Script file does not exist. "
                             f"You are trying to run {self.work_directory / self.script_location}.")

        self.pre_modules = pre_modules
        self.run_modules = run_modules

        self.additional_script = additional_scripts

    def load_modules(self):
        run(["module", "purge"])
        for module in self.pre_modules:
            run(["module", "load", module])

    def _get_environment(self):
        if self.python_environment is not None and self._use_conda:
            return f"conda activate {self.python_environment}"
        elif self.python_environment is not None:
            return f"source {self.python_environment / 'bin/activate'}"
        return ""

    def source_environment(self):
        if self.python_environment is not None:
            run(["echo", f"Activate environment {str(self.python_environment)}"])
            run(self._get_environment())

    def setup_experiment(self):
        if self._setup_experiment:
            if (self.work_directory / "setup.py").exists():
                run(["pip", "install", "-e", str(self.work_directory)])
            elif (self.work_directory / "requirements.txt").exists():
                run(["pip", "install", "-r", str(self.work_directory / "requirements.txt")])
            if (self.work_directory / "offline_setup.py").exists():
                run(["python", str(self.work_directory / "offline_setup")])

    def new_run(self):
        self.source_environment()
        self.load_modules()
        if self.additional_script is not None:
            for additional_install in self.additional_script:
                run(additional_install)
        self.setup_experiment()

        commands = [
            "jobId=$SLURM_JOB_ID",
            f"runWorkdirJob={str(self.run_work_directory)}/$jobId",
        ]

        commands.extend([
            'mkdir -p "$runWorkdirJob"',
            'mkdir "$runWorkdirJob/checkpoints"',

            f'cp -r {str(self.work_directory)} $runWorkdirJob',
            f'cd "$runWorkdirJob/{self.work_directory.resolve().name}/{str(self.script_location.parent)}"',
            'module purge'
        ])
        commands.extend([f'module load {module}' for module in self.run_modules])
        if self.python_environment is not None:
            commands.extend([
                self._get_environment(),
            ])
        return commands
