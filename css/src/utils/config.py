import os
from typing import Any, Dict

import yaml


class Config:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, item: str) -> Any:
        v = self._data.get(item)
        if isinstance(v, dict):
            return Config(v)
        return v

    def __getitem__(self, item: str) -> Any:
        v = self._data[item]
        if isinstance(v, dict):
            return Config(v)
        return v

    def to_dict(self) -> Dict[str, Any]:
        return self._data


def load_config(path: str) -> Config:
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data)
