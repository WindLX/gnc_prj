from dataclasses import dataclass

import numpy as np


@dataclass
class Vector3:
    x: float
    y: float
    z: float

    @staticmethod
    def from_list(lst: list | tuple | np.ndarray) -> "Vector3":
        return Vector3(float(lst[0]), float(lst[1]), float(lst[2]))

    @staticmethod
    def zero() -> "Vector3":
        return Vector3(0, 0, 0)

    def list(self) -> list:
        return [self.x, self.y, self.z]

    def numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, index: int, value: float) -> None:
        if index == 0:
            self.x = float(value)
        elif index == 1:
            self.y = float(value)
        elif index == 2:
            self.z = float(value)
        else:
            raise IndexError("Index out of range")

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __str__(self) -> str:
        return f"[{self.x}, {self.y}, {self.z}]"

    def __repr__(self) -> str:
        return f"[{self.x}, {self.y}, {self.z}]"
