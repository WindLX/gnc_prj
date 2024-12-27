import os
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv

from model import Vector3


@dataclass
class PathStyle:
    weight: int
    color: str
    transparency: float
    fillcolor: str
    fillTransparency: float


class Map:
    def __init__(self, key: Optional[str] = None):
        if key == None:
            load_dotenv()
            self.key = os.getenv("key")
        else:
            self.key = key

    def construct_paths_uri(self, paths: list[tuple[list[Vector3], PathStyle]]) -> str:
        paths_str_list = []

        for locations, style in paths:
            path_str = f"{style.weight},{style.color},{style.transparency},{style.fillcolor},{style.fillTransparency}:"
            locations_str = ";".join([f"{loc.x},{loc.y}" for loc in locations])
            paths_str_list.append(path_str + locations_str)

        paths_str = "|".join(paths_str_list)

        uri = f"https://restapi.amap.com/v3/staticmap?zoom=15&size=500*500&paths={paths_str}&key={self.key}"
        return uri

    def get_map_image(self, uri: str):
        response = requests.get(uri)
        return response.content

    def save_map_image(self, uri: str, filename: str):
        response = requests.get(uri)
        with open(filename, "wb") as f:
            f.write(response.content)


if __name__ == "__main__":
    locations = [
        Vector3(116.31604, 39.96491, 0),
        Vector3(116.320816, 39.966606, 0),
        Vector3(116.321785, 39.966827, 0),
        Vector3(116.32361, 39.966957, 0),
    ]
    path_styles = [
        PathStyle(
            weight=5,
            color="0x0000FF",
            transparency=1,
            fillcolor="",
            fillTransparency=0.5,
        )
    ]
    map = Map()
    uri = map.construct_paths_uri([(locations, path_styles[0])])
    print(uri)
    map.save_map_image(uri, "map.png")
