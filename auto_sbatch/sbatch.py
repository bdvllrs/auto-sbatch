import subprocess
from itertools import product
from pathlib import Path
from typing import List

from omegaconf import DictConfig, ListConfig, OmegaConf

from auto_sbatch.experiment_handler import ExperimentHandler
from auto_sbatch.processes import Command

class SBatch:
    def __init__(
            self,
            slurm_params=None,
            params=None,
            grid_search=None,
            run_script=None,
            experiment_handler: ExperimentHandler = None
    ):
        self._experiment_handler = experiment_handler
        self._slurm_params = OmegaConf.create(slurm_params or dict())
        self._params = OmegaConf.create(params or dict())
        self._commands: List[Command] = []
        self._n_job_seq = 1
        self._main_command_args = {}

        self._grid_search = grid_search
        self._run_script = run_script

        self.set_grid_search()

        if self._experiment_handler is None and self._run_script is None:
            raise ValueError(f"Missing run_script param")

        if self._experiment_handler is not None:
            self.add_commands(self._experiment_handler.new_run())
            self._main_command_args.update(self._experiment_handler.get_main_command_args())

    @property
    def num_available_jobs(self):
        return self._n_job_seq

    def get_task_params(self, task_id=0):
        dot_params = walk_dict(self._params)
        params = {}
        for key, value in walk_dict(dot_params).items():
            if (self._grid_search is not None and
                    key in self._grid_search and isinstance(value, ListConfig)):
                key_var = key.replace(".", "").replace("/", "")
                params[key_var] = value[task_id]
            else:
                params[key] = value
        return params

    def set_grid_search(self):
        n_jobs = None
        if self._grid_search is not None:
            n_jobs, self._params = get_grid_combinations(self._params, self._grid_search)
            self._n_job_seq = n_jobs
        if "--array" in self._slurm_params and self._slurm_params["--array"] == "auto":
            assert n_jobs is not None, "Cannot have --array=auto when no grid_search is set."
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
        param_end = "_param"
        if "--array" in self._slurm_params:
            param_end += "[$taskId]"
        for key, value in dot_params.items():
            if (self._grid_search is not None and
                    key in self._grid_search and isinstance(value, ListConfig)):
                key_var = key.replace(".", "_").replace("/", "_")
                s = ' "' + key + '=${' + key_var + param_end + '}"'
                params["grid_search"] += s
                params["all"] += s
                params["grid_search_string"].append(key + '=${' + key_var + param_end + '}')
            else:
                escaped_value = str(value).replace('"', '\\"')
                s = f' "{key}={escaped_value}"'
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

    def make_slurm_script(self, run_command, task_id=None, main_command_args=None):
        run_command = Command(run_command)
        dot_params_slurm = walk_dict(self._slurm_params)
        dot_params = walk_dict(self._params)
        script_name = self._run_script if self._run_script is not None else self._experiment_handler.script_location.name

        slurm_script = "#!/bin/sh"

        for key, value in dot_params_slurm.items():
            if key[:2] == "--":
                slurm_script += f"\n#SBATCH {key}={value}"
            elif key[:1] == "-":
                slurm_script += f"\n#SBATCH {key} {value}"
        slurm_script += "\n"
        for command in self._commands:
            slurm_script += f"\n{command.get()}"

        for key, param_values in walk_dict(dot_params).items():
            if (self._grid_search is not None and
                    key in self._grid_search and
                    isinstance(param_values, ListConfig) and
                    '--array' not in dot_params_slurm and task_id is not None):
                key_var = key.replace(".", "_").replace("/", "_")
                slurm_script += f"\n{key_var}_param={param_values[task_id]}"
            elif (self._grid_search is not None and
                  key in self._grid_search and isinstance(param_values, ListConfig)):
                key_var = key.replace(".", "_").replace("/", "_")
                slurm_script += '\n' + key_var + '_param=("' + '" "'.join(map(str, param_values)) + '")'

        if '--array' in dot_params_slurm:
            slurm_script += f'\ntaskId=$SLURM_ARRAY_TASK_ID'
        elif task_id is None and self._n_job_seq > 1:
            slurm_script += f'\nfor taskId in $(seq 0 {self._n_job_seq - 1})\ndo'
        elif task_id is None:
            slurm_script += f'\ntaskId=0'

        run_command_params = self.get_run_params()
        run_command_args = {
            "script_name": script_name,
            "num_gpus": self.get_num_gpus(),
            "params": run_command_params["params"],
            "grid_search_params": run_command_params["grid_search"],
            "grid_search_string": run_command_params["grid_search_string"],
            "all_params": run_command_params["all"],
        }
        run_command_args.update(self._main_command_args)
        run_command_args.update(main_command_args or {})

        run_command.format(**run_command_args)
        slurm_script += f"\n{run_command.get()}"

        if self._n_job_seq > 1 and task_id is None and '--array' not in dot_params_slurm:
            slurm_script += f'\ndone'

        return slurm_script

    def __call__(
            self, run_command, task_id=None, schedule_all_tasks=False, save_script=None,
            run_script=True, main_command_args=None
    ):
        task_ids = [task_id]
        if schedule_all_tasks and "--array" not in self._slurm_params:
            task_ids = list(range(self._n_job_seq))
        for task_id in task_ids:
            slurm_script = self.make_slurm_script(run_command, task_id, main_command_args)
            if save_script is not None:
                path_location = Path(save_script)
                if task_id is not None:
                    path_location = path_location.with_name(path_location.name + "_" + task_id)
                with open(path_location, "w") as f:
                    f.write(slurm_script)
            if run_script:
                process = subprocess.Popen(
                    ["sbatch"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                (out, err) = process.communicate(bytes(slurm_script, 'utf-8'))
                print(slurm_script)
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


def get_grid_combinations(args, grid_search):
    cond = lambda key, val: key in grid_search and isinstance(val, ListConfig)
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


def auto_sbatch(run_command,
        slurm_params=None,
        params=None,
        grid_search=None,
        run_script=None,
        experiment_handler: ExperimentHandler = None):
    SBatch(slurm_params, params, grid_search, run_script, experiment_handler)(run_command)
