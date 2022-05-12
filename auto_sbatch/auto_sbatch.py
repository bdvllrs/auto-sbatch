import os
import subprocess
from itertools import product

from omegaconf import OmegaConf, DictConfig, ListConfig

from auto_sbatch.utils import ExperimentHandler

default_auto_sbatch_conf = {
    "-J": "run",
    "-N": 1,
    "-o": f"{os.getenv('SLURM_OUTPUT_LOG_DIR', '..')}/%j_out.log",
    "-e": f"{os.getenv('SLURM_OUTPUT_LOG_DIR', '..')}/%j_err.log",
    "python_environment": "???",
    "work_directory": ".",
    "run_work_directory": ".",
    "script_location": "???"
}


class SBatch:
    def __init__(self, experiment_handler: ExperimentHandler, sbatch_params=None):
        self.experiment_handler = experiment_handler
        self.slurm_script = "#!/bin/sh"
        self._params = sbatch_params
        self._available_slurm_commands = ["-J", "-N", "-n", "-o", "-e", "--gres", "--mem",
                                          "--time", "--mail-user", "--mail-type", "--array"]
        self._reserved_args = ["python_environment", "work_directory", "run_work_directory", "script_location",
                               "--grid-search"]
        self._commands = []

        self.add_commands(self.experiment_handler.new_run())

    def get_num_gpus(self):
        if '--gres' in self._params:
            s = self._params['--gres'].split(":")
            if len(s) > 1:
                return int(s[1])
        return 0

    def add_command(self, command):
        self._commands.append(command)

    def add_commands(self, commands):
        for command in commands:
            self.add_command(command)

    def make_slurm_script(self):
        dot_params = walk_dict(self._params)
        for key, value in dot_params.items():
            if key in self._available_slurm_commands:
                if key[:2] == "--":
                    self.slurm_script += f"\n#SBATCH {key}={value}"
                elif key[:1] == "-":
                    self.slurm_script += f"\n#SBATCH {key} {value}"
        self.slurm_script += "\n"
        for command in self._commands:
            self.slurm_script += f"\n{command}"

        for key, value in walk_dict(dot_params).items():
            if key not in self._available_slurm_commands and key not in self._reserved_args:
                if ("--grid-search" in dot_params and
                        key in dot_params["--grid-search"] and isinstance(value, ListConfig)):
                    key_var = key.replace(".", "").replace("/", "")
                    self.slurm_script += '\n' + key_var + 'Param=("' + '" "'.join(map(str, value)) + '")'

        self.slurm_script += f'\ntaskId=' + ("$SLURM_ARRAY_TASK_ID" if '--array' in dot_params else "0")

        self.slurm_script += f'\npython "{self.experiment_handler.script_location.name}" ' \
                             f'"gpus={self.get_num_gpus()}"'
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
        out, err = b"", b""
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


def auto_sbatch(arg_config=None):
    conf = OmegaConf.create(default_auto_sbatch_conf)
    if arg_config is not None:
        conf = OmegaConf.merge(conf, arg_config)
    cli_args = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_args)

    if "--grid-search" in conf and "--array" not in conf:
        print("Detected Grid-Search, adding --array=auto option for you.")
        conf["--array"] = "auto"

    if "--array" in conf:
        n_jobs = None
        if "--grid-search" in conf:
            n_jobs, conf = get_grid_combinations(conf)
        if conf["--array"] == "auto":
            assert n_jobs is not None, "Cannot have --array=auto when no grid-search is set."
            conf["--array"] = f"0-{n_jobs - 1}"

            if n_jobs == 1:
                del conf["--array"]

    handler = ExperimentHandler(
        conf.python_environment,
        conf.work_directory,
        conf.run_work_directory,
        conf.script_location,
        "--array" in conf
    )

    sbatch = SBatch(handler, conf)
    sbatch()


if __name__ == '__main__':
    auto_sbatch()
