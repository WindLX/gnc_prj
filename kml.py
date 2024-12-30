import re
import bisect
from dataclasses import dataclass
from datetime import datetime, timedelta

from model import Vector3
from utils import bias


@dataclass
class TrackData:
    when: datetime
    coord: Vector3


class KML:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.track_data: list[TrackData] = []

    def parse(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        when_pattern = re.compile(r"<when>(.*?)</when>")
        coord_pattern = re.compile(r"<coord>(.*?)</coord>")

        current_when = None

        for line in lines:
            when_match = when_pattern.search(line)
            if when_match:
                current_when = when_match.group(1).strip()
                current_when = datetime.fromisoformat(current_when).replace(tzinfo=None)
                current_when = current_when - timedelta(hours=8)

            coord_match = coord_pattern.search(line)
            if coord_match and current_when:
                coord_data = coord_match.group(1).strip().split()
                if len(coord_data) == 3:
                    coord = Vector3(
                        float(coord_data[1]), float(coord_data[0]), float(coord_data[2])
                    )
                    coord[0] = coord[0] + bias[0]
                    coord[1] = coord[1] + bias[1]
                    track = TrackData(when=current_when, coord=coord)
                    self.track_data.append(track)
                    current_when = None

    def get_track_data(self) -> list[TrackData]:
        return self.track_data

    def interpolate_at(self, target_time: datetime) -> Vector3:
        times = [track.when for track in self.track_data]
        idx = bisect.bisect_left(times, target_time)

        if idx == 0:
            return self.track_data[0].coord
        if idx == len(self.track_data):
            return self.track_data[-1].coord

        track1 = self.track_data[idx - 1]
        track2 = self.track_data[idx]

        return Vector3.interpolate(
            track1.coord, track2.coord, track1.when, track2.when, target_time
        )


if __name__ == "__main__":
    kml_file_path = "./data/1_Medium_Interference_Near_SAA_1525/doc.kml"
    kml = KML(kml_file_path)
    kml.parse()

    for track in kml.get_track_data():
        print(f"when: {track.when}, coord: {track.coord}")

    target_time = datetime(2024, 12, 14, 7, 17, 57, 0)

    try:
        interpolated_coord = kml.interpolate_at(target_time)
        print(
            f"Interpolated coord at {target_time}: ({interpolated_coord.x}, {interpolated_coord.y}, {interpolated_coord.z})"
        )
    except ValueError as e:
        print(e)
