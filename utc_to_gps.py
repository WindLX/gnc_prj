import numpy as np
from datetime import datetime, timedelta

from julian_day import julian_day
from leap_seconds import leap_seconds
from gps_constants import GpsConstants


def check_utc_time_inputs(utc_time):
    if utc_time.shape[1] != 6:
        raise ValueError("utcTime must have 6 columns")
    if not np.all(np.mod(utc_time[:, :3], 1) == 0):
        raise ValueError("year, month & day must be integers")
    if np.any(utc_time[:, 0] < 1980) or np.any(utc_time[:, 0] > 2099):
        raise ValueError("year must have values in the range: [1980:2099]")
    if np.any(utc_time[:, 1] < 1) or np.any(utc_time[:, 1] > 12):
        raise ValueError("The month in utcTime must be a number in the set [1:12]")
    if np.any(utc_time[:, 2] < 1) or np.any(utc_time[:, 2] > 31):
        raise ValueError("The day in utcTime must be a number in the set [1:31]")
    if np.any(utc_time[:, 3] < 0) or np.any(utc_time[:, 3] >= 24):
        raise ValueError("The hour in utcTime must be in the range [0,24)")
    if np.any(utc_time[:, 4] < 0) or np.any(utc_time[:, 4] >= 60):
        raise ValueError("The minutes in utcTime must be in the range [0,60)")
    if np.any(utc_time[:, 5] < 0) or np.any(utc_time[:, 5] > 60):
        raise ValueError("The seconds in utcTime must be in the range [0,60]")
    return True


def utc_to_gps(utc_time):
    check_utc_time_inputs(utc_time)

    HOURSEC = 3600
    MINSEC = 60
    WEEKSEC = GpsConstants.WEEKSEC
    DAYSEC = GpsConstants.DAYSEC

    days_since_epoch = np.floor(julian_day(utc_time) - GpsConstants.GPSEPOCHJD)

    gps_week = np.floor(days_since_epoch / 7).astype(int)
    day_of_week = np.mod(days_since_epoch, 7)

    gps_seconds = (
        day_of_week * DAYSEC
        + utc_time[:, 3] * HOURSEC
        + utc_time[:, 4] * MINSEC
        + utc_time[:, 5]
    )
    gps_week += np.floor(gps_seconds / WEEKSEC).astype(int)
    gps_seconds = np.mod(gps_seconds, WEEKSEC)

    leap_secs = leap_seconds(utc_time)
    fct_seconds = gps_week * WEEKSEC + gps_seconds + leap_secs

    gps_week = np.floor(fct_seconds / WEEKSEC).astype(int)
    gps_seconds = np.where(
        gps_week == 0, fct_seconds, np.mod(fct_seconds, gps_week * WEEKSEC)
    )

    gps_time = np.vstack((gps_week, gps_seconds)).T
    assert np.all(
        fct_seconds == gps_week * WEEKSEC + gps_seconds
    ), "Error in computing gpsWeek, gpsSeconds"

    return gps_time, fct_seconds


# Example usage:
# utc_time = np.array([[2023, 10, 1, 12, 0, 0]])
# gps_time, fct_seconds = utc_to_gps(utc_time)
# print(gps_time, fct_seconds)
