from auto_sbatch.slurm_script import SlurmScriptParser


def test_slurm_script_parser():
    main_command = 'python main.py "script_param=7" another_param="a"'
    slurm_script = """#!/bin/bash
#SBATCH --job-name=job-name
#SBATCH --time=01:00:00
#SBATCH -o job-name.out
#SBATCH --error=job-name.err
"""
    slurm_script += f"\n{main_command}"

    parser = SlurmScriptParser(
        slurm_script, main_command="python {script_name} {all_params}"
    )
    parser.parse()

    assert "--job-name" in parser.slurm_params
    assert "--time" in parser.slurm_params
    assert "-o" in parser.slurm_params
    assert "--error" in parser.slurm_params

    assert parser.slurm_params["--job-name"] == "job-name"
    assert parser.slurm_params["--time"] == "01:00:00"
    assert parser.slurm_params["-o"] == "job-name.out"
    assert parser.slurm_params["--error"] == "job-name.err"

    assert parser.main_command == main_command
    assert parser.script_name == "main.py"
    assert parser.params is not None
    assert "script_param" in parser.params.keys()
    assert "another_param" in parser.params.keys()
    assert parser.params.script_param == 7
    assert parser.params.another_param == "a"
