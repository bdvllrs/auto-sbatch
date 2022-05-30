import subprocess
from itertools import product
from typing import List

from omegaconf import OmegaConf, DictConfig, ListConfig

from auto_sbatch.experiment_handler import ExperimentHandler
from auto_sbatch.processes import Command

default_auto_sbatch_conf = {
    "slurm": {
      "-J": "run",
      "-N": 1
    },
    "python_environment": "???",
    "work_directory": ".",
    "run_work_directory": ".",
    "script_location": "???",
    "run_command": 'python "{script_name}" "num_gpus={num_gpus}" {all_params} "checkpoints_dir=\'{checkpoints_dir}\'"'
}


class SBatch:
    def __init__(self, slurm_params=None, sbatch_params=None, experiment_handler: ExperimentHandler = None):
        self.experiment_handler = experiment_handler
        self.slurm_script = "#!/bin/sh"
        self._slurm_params = OmegaConf.create(slurm_params if slurm_params is not None else dict())
        self._params = OmegaConf.create(sbatch_params if sbatch_params is not None else dict())
        self._reserved_args = ["python_environment", "work_directory", "run_work_directory", "script_location",
                               "--grid-search", "run_registry_path"]
        self._commands: List[Command] = []

        self.set_grid_search()

        if self.experiment_handler is None and "--run-script" not in self._params:
            raise ValueError(f"Missing --run-script param")

        if self.experiment_handler is not None:
            self.add_commands(self.experiment_handler.new_run())

    def set_grid_search(self):
        if "--grid-search" in self._params and "--array" not in self._slurm_params:
            print("Detected Grid-Search, adding --array=auto option for you.")
            self._slurm_params["--array"] = "auto"

        if "--array" in self._slurm_params:
            n_jobs = None
            if "--grid-search" in self._params:
                n_jobs, self._params = get_grid_combinations(self._params)
            if self._slurm_params["--array"] == "auto":
                assert n_jobs is not None, "Cannot have --array=auto when no grid-search is set."
                self._slurm_params["--array"] = f"0-{n_jobs - 1}"

                if n_jobs == 1:
                    del self._slurm_params["--array"]

    def get_num_gpus(self):
        if '--gres' in self._slurm_params:
            s = self._slurm_params['--gres'].split(":")
            if len(s) > 1:
                return int(s[1])
        return 0

    def get_run_params(self):
        params = {
            "params": "",
            "grid_search": "",
            "all": "",
            "grid_search_string": []
        }
        dot_params = walk_dict(self._params)
        for key, value in dot_params.items():
            if key not in self._reserved_args:
                if ("--grid-search" in dot_params and
                        key in dot_params["--grid-search"] and isinstance(value, ListConfig)):
                    key_var = key.replace(".", "").replace("/", "")
                    s = ' "' + key + '=${' + key_var + 'Param[$taskId]}"'
                    params["grid_search"] += s
                    params["all"] += s
                    params["grid_search_string"].append(key + '=${' + key_var + 'Param[$taskId]}')
                else:
                    s = f' "{key}={value}"'
                    params["params"] += s
                    params["all"] += s
        params["grid_search_string"] = "_".join(params["grid_search_string"])
        return params

    def add_command(self, command):
        if not isinstance(command, Command):
            command = Command(command)
        self._commands.append(command)

    def add_commands(self, commands):
        for command in commands:
            self.add_command(command)

    def make_slurm_script(self, run_command):
        run_command = Command(run_command)
        dot_params_slurm = walk_dict(self._slurm_params)
        dot_params = walk_dict(self._params)
        script_name = self._params[
            "--run-script"] if "--run-script" in self._params else self.experiment_handler.script_location.name

        for key, value in dot_params_slurm.items():
            if key[:2] == "--":
                self.slurm_script += f"\n#SBATCH {key}={value}"
            elif key[:1] == "-":
                self.slurm_script += f"\n#SBATCH {key} {value}"
        self.slurm_script += "\n"
        for command in self._commands:
            self.slurm_script += f"\n{command.get()}"

        for key, value in walk_dict(dot_params).items():
            if key not in self._reserved_args:
                if ("--grid-search" in dot_params and
                        key in dot_params["--grid-search"] and isinstance(value, ListConfig)):
                    key_var = key.replace(".", "").replace("/", "")
                    self.slurm_script += '\n' + key_var + 'Param=("' + '" "'.join(map(str, value)) + '")'

        self.slurm_script += f'\ntaskId=' + ("$SLURM_ARRAY_TASK_ID" if '--array' in dot_params else "0")

        run_command_params = self.get_run_params()
        run_command_args = {
            "script_name": script_name,
            "num_gpus": self.get_num_gpus(),
            "params": run_command_params["params"],
            "grid_search_params": run_command_params["grid_search"],
            "grid_search_string": run_command_params["grid_search_string"],
            "all_params": run_command_params["all"],
            "checkpoints_dir": "../../checkpoints/$jobId"
        }

        run_command.format(**run_command_args)
        self.slurm_script += f"\n{run_command.get()}"

    def __call__(self, run_command):
        self.make_slurm_script(run_command)
        process = subprocess.Popen(["sbatch"],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        (out, err) = process.communicate(bytes(self.slurm_script, 'utf-8'))
        # (out, err) = b"", b""
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


def auto_sbatch(run_command, slurm_config=None, arg_config=None, handler=None):
    SBatch(slurm_config, arg_config, handler)(run_command)


def main(arg_config=None):
    conf = OmegaConf.create(default_auto_sbatch_conf)
    if arg_config is not None:
        arg_config = OmegaConf.create(arg_config)
        conf = OmegaConf.merge(conf, arg_config)
    cli_args = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_args)
    slurm_conf = conf.slurm
    run_command = conf.run_command
    del conf.slurm
    del conf.run_command

    handler = None
    if "python_environment" in conf:
        handler = ExperimentHandler(
            conf.python_environment,
            conf.work_directory,
            conf.run_work_directory,
            conf.script_location
        )
    auto_sbatch(run_command, slurm_conf, conf, handler)


if __name__ == '__main__':
    main()
