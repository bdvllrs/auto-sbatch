from pathlib import Path

from auto_sbatch.processes import run, Command


class ExperimentHandler:
    def __init__(self,
                 script_location,
                 work_directory=".",
                 run_work_directory=".",
                 python_environment=None,
                 run_registry_path=None,
                 pre_modules=None,
                 run_modules=None,
                 additional_scripts=None):
        self.script_location = Path(script_location)
        self.run_work_directory = Path(run_work_directory)
        self.work_directory = Path(work_directory)
        self.python_environment = Path(python_environment) if python_environment is not None else None
        self.run_registry_path = Path(run_registry_path) if run_registry_path is not None else None

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

    def source_environment(self):
        if self.python_environment is not None:
            environment = str(self.python_environment / "bin/activate")
            run(["echo", f"Activate environment {environment}"])
            run(["source", environment])

    def setup_experiment(self):
        if (self.work_directory / "setup.py").exists():
            run(["pip", "install", "-e", str(self.work_directory)])
        elif (self.work_directory / "requirements.txt").exists():
            run(["pip", "install", "-r", str(self.work_directory / "requirements.txt")])
        if (self.work_directory / "offline_setup.py").exists():
            run(["python", str(self.work_directory / "offline_setup")])

    def new_run(self):
        self.source_environment()
        self.load_modules()
        for additional_install in self.additional_script:
            run(additional_install)
        self.setup_experiment()

        # add_run_location = Path(__file__).parent / "register_run.py"
        commands = [
            "jobId=$SLURM_JOB_ID",
            f"runWorkdirJob={str(self.run_work_directory)}/$jobId",
        ]

        commands.extend([
            'mkdir -p "$runWorkdirJob"',
            'mkdir "$runWorkdirJob/logs"',
            'mkdir "$runWorkdirJob/checkpoints"',

            f'cp -r {str(self.work_directory)} $runWorkdirJob',
            f'cd "$runWorkdirJob/{self.work_directory.absolute().name}/{str(self.script_location.parent)}"',
            'module purge'
        ])
        commands.extend([f'module load {module}' for module in self.run_modules])
        if self.python_environment is not None:
            commands.extend([
                f'source "{str(self.python_environment)}/bin/activate"'
            ])
        if self.run_registry_path is not None:
            commands.append(
                f'register-run registry={str(self.run_registry_path.absolute())} '
                f'job_id=$jobId status="started" location="$runWorkdirJob/{self.work_directory.absolute().name}/{str(self.script_location)}"',
            )
        return commands
