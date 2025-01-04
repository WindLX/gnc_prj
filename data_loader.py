from dataclasses import dataclass
from pathlib import Path

import toml


@dataclass
class Trajectory:
    nav: Path
    obs: Path
    kml: Path

    def unpack(self) -> tuple[Path, Path, Path]:
        return self.nav, self.obs, self.kml


class DataLoader:
    def __init__(self, directory: Path | str):
        if isinstance(directory, str):
            directory = Path(directory)
        self.directory = directory
        self.index_file = self.directory / "index.toml"
        self.observations = self._load_observations()

    def _load_observations(self) -> list[Trajectory]:
        with open(self.index_file, "r") as file:
            data = toml.load(file)
        observations = []
        for item in data.values():
            for entry in item:
                observations.append(
                    Trajectory(
                        nav=self.directory / Path(entry["nav"]),
                        obs=self.directory / Path(entry["obs"]),
                        kml=self.directory / Path(entry["kml"]),
                    )
                )
        return observations

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx]
