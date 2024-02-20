import subprocess


class Command:
    def __init__(self, command: "str | Command"):
        if isinstance(command, Command):
            command = command.get()
        assert isinstance(command, str)
        self.command = command

    def get(self) -> str:
        return self.command

    def format(self, *args, **kwargs):
        self.command = self.command.format(*args, **kwargs)


class Python(Command):
    def get(self) -> str:
        return "python << EOF\n" + self.command + "\nEOF"


def run(command: str | Command):
    if isinstance(command, str):
        command = Command(command)
    if isinstance(command, list):
        command = Command(" ".join(command))
    subprocess.run(command.get(), shell=True)
