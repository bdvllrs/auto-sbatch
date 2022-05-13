import subprocess
import pytest
import auto_sbatch


def mock_run(command):
    if isinstance(command, str):
        command = auto_sbatch.processes.Command(command)
    if isinstance(command, list):
        command = auto_sbatch.processes.Command(" ".join(command))
    print(command.get())


class MockedPopen:
    @staticmethod
    def communicate(*args, **kwargs):
        return b"Mocked communication output", b"Mocked communication error"


def mock_subprocess_popen(*args, **kwargs):
    return MockedPopen()


@pytest.fixture
def mock_processes(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", mock_subprocess_popen)
    # monkeypatch.setattr("auto_sbatch.processes.run", mock_run)
