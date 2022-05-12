import csv
from pathlib import Path

from omegaconf import OmegaConf


def register_run():
    default_args = {
        "registry": "???",
        "job_id": "???",
        "status": "???",
        "location": "???"
    }
    conf = OmegaConf.create(default_args)
    cli_args = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_args)

    path = Path(conf.registry)
    if not path.exists():
        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["jobId", "status", "location"])

    with open(path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([conf.job_id, conf.status, conf.location])


if __name__ == '__main__':
    register_run()
