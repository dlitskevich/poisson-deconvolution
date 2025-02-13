import json
import os


class SearchConfig:
    @staticmethod
    def from_path(path: str):
        with open(path, "r") as file:
            spec = json.load(file)
        return SearchConfig.from_json(spec)

    @staticmethod
    def from_json(spec: dict):
        return SearchConfig(
            scales=spec["scales"],
            init_guesses=spec["init_guesses"],
        )

    def __init__(
        self,
        scales: list[float],
        init_guesses: list[int],
    ):
        self.scales = scales
        self.init_guesses = init_guesses

    def to_json(self) -> dict:
        return {
            "scales": self.scales,
            "init_guesses": self.init_guesses,
        }

    def dump(self, path: str):
        with open(path, "w") as file:
            json.dump(self.to_json(), file)


def read_search_file(dir: str) -> SearchConfig:
    name = "search.json"
    try:
        config = SearchConfig.from_path(os.path.join(dir, name))
        print(f"Successfully read search config from {dir}")
        return config
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {name}")
