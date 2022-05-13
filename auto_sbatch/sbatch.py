import subprocess
from itertools import product
from typing import List

from omegaconf import OmegaConf, DictConfig, ListConfig

from auto_sbatch.experiment_handler import ExperimentHandler
from auto_sbatch.processes import Command, run

default_auto_sbatch_conf = {
    "-J": "run",
    "-N": 1,
    "python_environment": "???",
    "work_directory": ".",
    "run_work_directory": ".",
    "script_location": "???"
}


class SBatch:
    def __init__(self, sbatch_params=None, experiment_handler: ExperimentHandler = None):
        self.experiment_handler = experiment_handler
        self.slurm_script = "#!/bin/sh"
        self._params = OmegaConf.create(sbatch_params)
        self._available_slurm_commands = ["-J", "-N", "-n", "-o", "-e", "--gres", "--mem",
                                          "--time", "--mail-user", "--mail-type", "--array"]
        self._reserved_args = ["python_environment", "work_directory", "run_work_directory", "script_location",
                               "--grid-search", "run_registry_path"]
        self._commands: List[Command] = []

        self.set_grid_search()

        if self.experiment_handler is None and "--run-script" not in self._params:
            raise ValueError(f"Missing --run-script param")

        if self.experiment_handler is not None:
            self.add_commands(self.experiment_handler.new_run())

    def set_grid_search(self):
        if "--grid-search" in self._params and "--array" not in self._params:
            print("Detected Grid-Search, adding --array=auto option for you.")
            self._params["--array"] = "auto"

        if "--array" in self._params:
            n_jobs = None
            if "--grid-search" in self._params:
                n_jobs, self._params = get_grid_combinations(self._params)
            if self._params["--array"] == "auto":
                assert n_jobs is not None, "Cannot have --array=auto when no grid-search is set."
                self._params["--array"] = f"0-{n_jobs - 1}"

                if n_jobs == 1:
                    del self._params["--array"]

    def get_num_gpus(self):
        if '--gres' in self._params:
            s = self._params['--gres'].split(":")
            if len(s) > 1:
                return int(s[1])
        return 0

    def add_command(self, command):
        if not isinstance(command, Command):
            command = Command(command)
        self._commands.append(command)

    def add_commands(self, commands):
        for command in commands:
            self.add_command(command)

    def make_slurm_script(self):
        dot_params = walk_dict(self._params)
        script_name = self._params[
            "--run-script"] if "--run-script" in self._params else self.experiment_handler.script_location.name

        for key, value in dot_params.items():
            if key in self._available_slurm_commands:
                if key[:2] == "--":
                    self.slurm_script += f"\n#SBATCH {key}={value}"
                elif key[:1] == "-":
                    self.slurm_script += f"\n#SBATCH {key} {value}"
        self.slurm_script += "\n"
        for command in self._commands:
            self.slurm_script += f"\n{command.get()}"

        for key, value in walk_dict(dot_params).items():
            if key not in self._available_slurm_commands and key not in self._reserved_args:
                if ("--grid-search" in dot_params and
                        key in dot_params["--grid-search"] and isinstance(value, ListConfig)):
                    key_var = key.replace(".", "").replace("/", "")
                    self.slurm_script += '\n' + key_var + 'Param=("' + '" "'.join(map(str, value)) + '")'

        self.slurm_script += f'\ntaskId=' + ("$SLURM_ARRAY_TASK_ID" if '--array' in dot_params else "0")

        self.slurm_script += f'\npython "{script_name}" "gpus={self.get_num_gpus()}"'
        for key, value in walk_dict(dot_params).items():
            if key not in self._available_slurm_commands and key not in self._reserved_args:
                if ("--grid-search" in dot_params and
                        key in dot_params["--grid-search"] and isinstance(value, ListConfig)):
                    key_var = key.replace(".", "").replace("/", "")
                    self.slurm_script += ' "' + key + '=${' + key_var + 'Param[$taskId]}"'
                else:
                    self.slurm_script += f' "{key}={value}"'
        self.slurm_script += ' "checkpoints_dir=\'../../checkpoints/$jobId\'"'

    def __call__(self):
        self.make_slurm_script()
        process = subprocess.Popen(["sbatch"],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        (out, err) = process.communicate(bytes(self.slurm_script, 'utf-8'))
        print("Running generated SLURM script:")
        print(self.slurm_script)
        out, err = bytes.decode(out), bytes.decode(err)
        if len(out):
            print(out)
        if len(err):
            print(err)


def walk_dict(d, prefix=[], cond=None):
    new_dic = {}
    if d is not None:
        for key, val in d.items():
            if type(val) is DictConfig:
                new_dic.update(walk_dict(val, prefix + [key], cond))
            else:
                dotted_key = ".".join(prefix + [key])
                if cond is None or cond(dotted_key, val):
                    new_dic[dotted_key] = val
    return new_dic


def get_grid_combinations(args):
    cond = lambda key, val: key in args["--grid-search"] and isinstance(val, ListConfig)
    unstructured_dict = walk_dict(args, cond=cond)

    keys, values = zip(*unstructured_dict.items())
    all_combinations = list(product(*values))
    n_jobs = len(all_combinations)
    new_dict = [
        f"{key}=[{', '.join([str(all_combinations[i][k]) for i in range(len(all_combinations))])}]" for k, key in
        enumerate(keys)
    ]
    new_values = OmegaConf.from_dotlist(new_dict)
    return n_jobs, OmegaConf.merge(args, new_values)


def auto_sbatch(arg_config=None, handler=None):
    SBatch(arg_config, handler)()
    run("test")

def main(arg_config=None):
    conf = OmegaConf.create(default_auto_sbatch_conf)
    if arg_config is not None:
        arg_config = OmegaConf.create(arg_config)
        conf = OmegaConf.merge(conf, arg_config)
    cli_args = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_args)

    handler = None
    if "python_environment" in conf:
        handler = ExperimentHandler(
            conf.python_environment,
            conf.work_directory,
            conf.run_work_directory,
            conf.script_location
        )
    auto_sbatch(conf, handler)


if __name__ == '__main__':
    main()
