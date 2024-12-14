from dataclasses import dataclass
from datetime import datetime


GM = 3.986005e14  # WGS84 gravitational constant*earth mass
Omegae_dot = 7.2921151467e-5  # rad/s earth rotation rate


@dataclass
class SVData:
    x: float
    y: float


@dataclass
class ECEFData:
    x: float
    y: float
    z: float


@dataclass
class GPSTime:
    week: int
    sow: float
    dow: int


def date2gps(date: datetime) -> GPSTime:
    """
    Convert a calendar date to GPS time.

    Args:
        date (datetime): A datetime object representing the date in UTC.

    Returns:
        GPSTime: An object containing GPS week number, GPS seconds of the week, and GPS day of the week.
    """
    gps_start_date = datetime(1980, 1, 6)
    delta = date - gps_start_date

    gps_week = delta.days // 7
    gps_dow = delta.days % 7
    gps_sow = gps_dow * 86400 + date.hour * 3600 + date.minute * 60 + date.second

    return GPSTime(gps_week, gps_sow, gps_dow)


if __name__ == "__main__":
    date = datetime(2022, 10, 2, 0, 0, 0)
    gps_time = date2gps(date)
    print(gps_time)
