from collections.abc import Iterable, Mapping
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, List

from auto_sbatch import ExperimentHandler
from auto_sbatch.grid_search import GridSearch
from auto_sbatch.processes import Command
from auto_sbatch.slurm_script import SlurmScriptParser


class SBatch:
    def __init__(
        self,
        slurm_params: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        *,
        grid_search: GridSearch | None = None,
        script_name: str | None = None,
        experiment_handler: ExperimentHandler | None = None,
    ):
        self._slurm_params = slurm_params or {}
        self._params = params or {}
        self._commands: List[Command] = []
        self._post_commands: List[Command] = []
        self._n_job_seq = 1
        self._main_command_args: dict[str, Any] = {}

        self._grid_search = grid_search
        self._script_name = script_name

        if experiment_handler is not None:
            self.configure_from_experiment_handler(experiment_handler)

        self.set_grid_search()

    def configure_from_experiment_handler(self, handler: ExperimentHandler):
        self.add_commands(handler.new_run())
        self._main_command_args.update(handler.get_main_command_args())
        self.set_script_name(handler.script_location.name)

    def set_script_name(self, script_name: str):
        self._script_name = script_name

    @classmethod
    def from_slurm_script(cls, slurm_script: str, main_command: str) -> "SBatch":
        parser = SlurmScriptParser(slurm_script, main_command)
        parser.parse()
        sbatch = cls(parser.slurm_params, parser.params, script_name=parser.script_name)
        sbatch.add_commands(parser.commands)
        sbatch.add_commands(parser.post_commands, post=True)
        return sbatch

    @property
    def num_available_jobs(self) -> int:
        return self._n_job_seq

    def _is_grid_search_key(self, key, value):
        return (
            self._grid_search is not None
            and key in self._grid_search
            and isinstance(value, ListConfig)
        )

    def _is_slurm_array_auto(self) -> bool:
        return (
            "--array" in self._slurm_params and self._slurm_params["--array"] == "auto"
        )

    def get_task_params(self, task_id: int = 0):
        dot_params = get_dotlist_params(self._params)
        params = {}
        for key, value in dot_params.items():
            if self._is_grid_search_key(key, value):
                key_var = key.replace(".", "").replace("/", "")
                params[key_var] = value[task_id]
            else:
                params[key] = value
        return params

    def set_grid_search(self):
        n_jobs = None
        if self._grid_search is not None:
            n_jobs, self._params = self._grid_search.get_combinations(self._params)
            self._n_job_seq = n_jobs
        if self._is_slurm_array_auto():
            if n_jobs is None:
                raise ValueError("Cannot have --array=auto when no grid_search is set.")
            self._slurm_params["--array"] = f"0-{n_jobs - 1}"

            if n_jobs == 1:
                del self._slurm_params["--array"]

    def get_num_gpus(self) -> int:
        if "--gres" in self._slurm_params:
            s = self._slurm_params["--gres"].split(":")
            if len(s) > 1:
                return int(s[1])
        return 0

    def get_run_params(self) -> dict[str, str]:
        params = {"params": "", "grid_search": "", "all": "", "grid_search_string": ""}
        dot_params = get_dotlist_params(self._params)
        param_end = "_param"
        if "--array" in self._slurm_params:
            param_end += "[$taskId]"
        for key, value in dot_params.items():
            if self._is_grid_search_key(key, value):
                key_var = key.replace(".", "_").replace("/", "_")
                s = '"' + key + "=${" + key_var + param_end + '}"'
                params["grid_search"] += f" {s}"
                params["all"] += f" {s}"
                params["grid_search_string"] += (
                    "_" + key + "=${" + key_var + param_end + "}"
                )
            else:
                escaped_value = get_arg_value(value)
                s = f'"{key}={escaped_value}"'
                params["params"] += f" {s}"
                params["all"] += f" {s}"
        return params

    def add_command(self, command: str | Command, post: bool = False):
        command = Command(command)
        if post:
            self._post_commands.append(command)
            return
        self._commands.append(command)

    def add_commands(self, commands: Iterable[Command | str], post: bool = False):
        for command in commands:
            self.add_command(command, post)

    def make_slurm_script(
        self,
        run_command: str | Command,
        task_id: int | None = None,
        main_command_args=None,
    ):
        run_command = Command(run_command)
        dot_params_slurm = get_dotlist_params(self._slurm_params)
        dot_params = get_dotlist_params(self._params)
        script_name = self._script_name

        slurm_script = "#!/bin/sh"

        for key, value in dot_params_slurm.items():
            if key[:2] == "--":
                slurm_script += f"\n#SBATCH {key}={value}"
            elif key[:1] == "-":
                slurm_script += f"\n#SBATCH {key} {value}"
        slurm_script += "\n"
        for command in self._commands:
            slurm_script += f"\n{command.get()}"

        for key, param_values in dot_params.items():
            if (
                self._is_grid_search_key(key, param_values)
                and "--array" not in dot_params_slurm
                and task_id is not None
            ):
                key_var = key.replace(".", "_").replace("/", "_")
                formatted_value = get_arg_value(param_values[task_id])
                slurm_script += f"\n{key_var}_param={formatted_value}"
            elif self._is_grid_search_key(key, param_values):
                key_var = key.replace(".", "_").replace("/", "_")
                slurm_script += (
                    "\n"
                    + key_var
                    + '_param=("'
                    + '" "'.join(map(get_arg_value, param_values))
                    + '")'
                )

        if "--array" in dot_params_slurm:
            slurm_script += "\ntaskId=$SLURM_ARRAY_TASK_ID"
        elif task_id is None and self._n_job_seq > 1:
            slurm_script += "\nfor taskId in $(seq 0 "
            slurm_script += f"{self._n_job_seq - 1})\ndo"
        elif task_id is None:
            slurm_script += "\ntaskId=0"

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

        if (
            self._n_job_seq > 1
            and task_id is None
            and "--array" not in dot_params_slurm
        ):
            slurm_script += "\ndone"

        for command in self._post_commands:
            slurm_script += f"\n{command.get()}"

        return slurm_script

    def run(
        self,
        run_command,
        task_id=None,
        schedule_all_tasks=False,
        save_script=None,
        main_command_args=None,
    ):
        task_ids = [task_id]
        if schedule_all_tasks and "--array" not in self._slurm_params:
            task_ids = list(range(self._n_job_seq))
        for task_id in task_ids:
            slurm_script = self.make_slurm_script(
                run_command, task_id, main_command_args
            )
            if save_script is not None:
                path_location = Path(save_script)
                if task_id is not None:
                    path_location = path_location.with_name(
                        path_location.name + "_" + task_id
                    )
                with open(path_location, "w") as f:
                    f.write(slurm_script)
            run(slurm_script)


def get_arg_value(value):
    if value is None:
        return "null"
    return str(value).replace('"', '\\"')


def run(slurm_script):
    process = Popen(["sbatch"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    (out, err) = process.communicate(bytes(slurm_script, "utf-8"))
    print(slurm_script)
    out, err = bytes.decode(out), bytes.decode(err)
    if len(out):
        print(out)
    if len(err):
        print(err)
