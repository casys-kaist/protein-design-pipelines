from types import SimpleNamespace
from typing import Iterable, Mapping, Dict, List

class MockContainer:
    def __init__(self, name: str, logs: List[str] | None = None):
        self.id = name
        self.name = name
        self.status = "running"
        self.attrs = {"State": {"Health": {"Status": "healthy"}}}
        self._logs = logs or []

class MockClient:
    """Minimal Docker client mock streaming predefined logs."""

    def __init__(self, logs: Mapping[str, Iterable[str]]):
        # Convert to lists for multiple iteration
        self._logs: Dict[str, List[str]] = {
            name: list(lines)
            for name, lines in logs.items()
        }
        self._exec_id = 0
        self._exec_map: Dict[str, str] = {}
        self.containers = SimpleNamespace(get=self._get_container)
        self.api = SimpleNamespace(
            exec_create=self._exec_create,
            exec_start=self._exec_start,
            exec_inspect=self._exec_inspect,
        )

    def _get_container(self, name: str) -> MockContainer:
        return MockContainer(name, self._logs.get(name, []))

    def _exec_create(self, container_id: str, *args, **kwargs) -> Dict[str, str]:
        self._exec_id += 1
        exec_id = f"{container_id}_exec_{self._exec_id}"
        self._exec_map[exec_id] = container_id
        return {"Id": exec_id}

    def _exec_start(self, exec_id: str, stream: bool = True):
        container_id = self._exec_map[exec_id]
        for line in self._logs.get(container_id, []):
            yield line.encode()

    def _exec_inspect(self, exec_id: str) -> Dict[str, int]:
        return {"ExitCode": 0}

def from_env(log_files: Mapping[str, str]) -> MockClient:
    """Load log files for each container and return a :class:`MockClient`."""
    logs: Dict[str, List[str]] = {}
    for name, path in log_files.items():
        with open(path, "r", encoding="utf-8") as f:
            logs[name] = [line.rstrip("\n") for line in f]
    return MockClient(logs)
