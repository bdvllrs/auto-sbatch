import os
import subprocess
from itertools import product

from omegaconf import OmegaConf, DictConfig, ListConfig

from utils import ExperimentHandler


class SBatch:
    def __init__(self, experiment_handler: 'ExperimentHandler', sbatch_params=None):
        self.experiment_handler = experiment_handler
        self.slurm_script = "#!/bin/sh"
        self._params = sbatch_params
        self._available_slurm_commands = ["-J", "-N", "-n", "-o", "-e", "--gres", "--mem",
                                          "--time", "--mail-user", "--mail-type", "--array"]
        self._reserved_args = ["python_environment", "work_directory", "run_work_directory", "script_location"]
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
        if self._params is not None:
            for key, value in self._params.items():
                if key in self._available_slurm_commands:
                    if key[:2] == "--":
                        self.slurm_script += f"\n#SBATCH {key}={value}"
                    elif key[:1] == "-":
                        self.slurm_script += f"\n#SBATCH {key} {value}"
            self.slurm_script += "\n"
        for command in self._commands:
            self.slurm_script += f"\n{command}"

        for key, value in walk_dict(self._params).items():
            if key not in self._available_slurm_commands and key not in self._reserved_args:
                if isinstance(value, list) and "--array" in self._params:
                    self.slurm_script += '\n' + key + 'Param=("' + '" "'.join(map(str, value)) + '")'

        self.slurm_script += f'\npython "{self.experiment_handler.script_location.name}" ' \
                             f'"gpus={self.get_num_gpus()}"'
        for key, value in walk_dict(self._params).items():
            if key not in self._available_slurm_commands and key not in self._reserved_args:
                if isinstance(value, list) and "--array" in self._params:
                    self.slurm_script += ' "' + key + '=${' + key + 'Param[$SLURM_ARRAY_TASK_ID]}"'
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
        print(self.slurm_script)
        print("Out: ", str(out))
        print("Err: ", str(err))


def walk_dict(d, prefix=[], only_lists=False):
    new_dic = {}
    for key, val in d.items():
        if type(val) is DictConfig:
            new_dic.update(walk_dict(val, prefix + [key], only_lists))
        else:
            if not only_lists or isinstance(val, ListConfig):
                new_dic[".".join(prefix + [key])] = val
    return new_dic


def get_grid_combinations(args):
    unstructured_dict = walk_dict(args, only_lists=True)

    keys, values = zip(*unstructured_dict.items())
    all_combinations = list(product(*values))
    n_jobs = len(all_combinations)
    new_dict = [
        f"{key}=[{', '.join([str(all_combinations[i][k]) for i in range(len(all_combinations))])}]" for k, key in
        enumerate(keys)
    ]
    new_values = OmegaConf.from_dotlist(new_dict)
    return n_jobs, OmegaConf.merge(args, new_values)


def auto_sbatch():
    default_args = {
        "-J": "run",
        "-N": 1,
        "-o": f"{os.getenv('SLURM_OUTPUT_LOG_DIR', '..')}/%j_out.log",
        "-e": f"{os.getenv('SLURM_OUTPUT_LOG_DIR', '..')}/%j_err.log",
        "python_environment": "???",
        "work_directory": ".",
        "run_work_directory": ".",
        "script_location": "???"
    }
    conf = OmegaConf.create(default_args)
    cli_args = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_args)

    if "--array" in conf:
        n_jobs, conf = get_grid_combinations(conf)
        if conf["--array"] == "auto":
            conf["--array"] = f"0-{n_jobs - 1}"

    handler = ExperimentHandler(
        conf.python_environment,
        conf.work_directory,
        conf.run_work_directory,
        conf.script_location,
        "--array" in conf
    )

    sbatch = SBatch(handler, OmegaConf.to_container(conf))
    sbatch()


if __name__ == '__main__':
    auto_sbatch()
