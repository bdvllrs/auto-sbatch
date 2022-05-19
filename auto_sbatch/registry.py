import subprocess

import pandas as pd


class Registry:
    def __init__(self, register_path):
        self.registry = pd.read_csv(register_path)

    def update_status(self):
        running_jobs = self.registry[self.registry["status"] == "started"]
        for idx, row in running_jobs.iterrows():
            p = subprocess.Popen([
                "sacct", f"--job={row['jobId']}", "--format=state"
            ], stdout=subprocess.PIPE)
            out = p.communicate()[0].decode().split("\n")
            if "COMPLETED" in out[4]:
                self.registry.at[idx, "status"] = "completed"
