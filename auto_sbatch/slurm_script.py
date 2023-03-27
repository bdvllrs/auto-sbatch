import re
from typing import Any, Dict, Tuple

from omegaconf import OmegaConf

from auto_sbatch.processes import Command


class SlurmScriptParser:
    def __init__(self, slurm_script: str, main_command: str):
        self._slurm_script = slurm_script
        self._main_command = main_command
        self.slurm_params: Dict[str, Any] = {}
        self.commands = []
        self.post_commands = []
        self.main_command = None
        self.script_name = None
        self.params = None

    def _format_main_command(self):
        possible_formats = {
            "script_name": r"([^\s]*?)",
            "num_gpus": r"([^\s]*?)",
            "params": r"(.*)",
            "grid_search_params": r"(.*)",
            "grid_search_string": r"(.*)",
            "all_params": r"(.*)",
        }
        for key, val in possible_formats.items():
            self._main_command = self._main_command.replace(
                "{" + key + "}", val
            )

    def parse(self) -> None:
        script_lines = self._slurm_script.strip("\n").split("\n")
        script_lines = [line.strip() for line in script_lines]
        self._format_main_command()
        main_command = self._main_command
        has_main_command = False
        for line in script_lines:
            if line.startswith("#SBATCH"):
                key, val = self._parse_slurm_line(line)
                self.slurm_params[key] = val
            elif matches := re.match(main_command, line):
                self.main_command = line
                self.script_name = matches.group(1)
                dotlist = matches.group(2).split(" ")
                dotlist = [
                    match[1:-1] if match.startswith('"') else match
                    for match in dotlist
                ]
                self.params = OmegaConf.from_dotlist(dotlist)
                has_main_command = True
            elif not has_main_command:
                self.commands.append(Command(line))
            else:
                self.post_commands.append(Command(line))

    @staticmethod
    def _parse_slurm_line(line) -> Tuple[str, Any]:
        line = line.replace("#SBATCH", "").strip()
        if "=" in line:
            key, val = line.split("=")
        else:
            key, val = line.split(" ")
        key = key.strip()
        val = val.strip()
        return key, val
