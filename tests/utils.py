from unittest import mock as mock


def mock_run(command):
    print(command)


def mock_for_tests(*, p_open=None, subprocess=None):
    if subprocess is not None:
        subprocess_instance = mock.MagicMock()
        subprocess_instance.run.side_effect = mock_run
        subprocess.return_value = subprocess_instance
    if p_open is not None:
        p_open_instance = mock.MagicMock()
        p_open_instance.communicate.return_value = (
            b"Mocked communication output",
            b"Mocked communication error"
        )
        p_open.return_value = p_open_instance
