import numpy as np
from datetime import datetime
from gps_constants import GpsConstants
from julian_day import julian_day


def leap_seconds(utc_time):
    """
    Find the number of leap seconds since the GPS Epoch.

    Parameters:
    utc_time: [mx6] matrix
        utc_time[i,:] = [year, month, day, hours, minutes, seconds]
        year must be specified using four digits, e.g. 1994
        year valid range: 1980 <= year <= 2099

    Output:
    leap_secs: array
        leap_secs[i] = number of leap seconds between the GPS Epoch and utc_time[i,:]
    """
    utc_time = np.array(utc_time)
    m, n = utc_time.shape
    if n != 6:
        raise ValueError("utc_time input must have 6 columns")

    utc_table = np.array(
        [
            [1982, 1, 1, 0, 0, 0],
            [1982, 7, 1, 0, 0, 0],
            [1983, 7, 1, 0, 0, 0],
            [1985, 7, 1, 0, 0, 0],
            [1988, 1, 1, 0, 0, 0],
            [1990, 1, 1, 0, 0, 0],
            [1991, 1, 1, 0, 0, 0],
            [1992, 7, 1, 0, 0, 0],
            [1993, 7, 1, 0, 0, 0],
            [1994, 7, 1, 0, 0, 0],
            [1996, 1, 1, 0, 0, 0],
            [1997, 7, 1, 0, 0, 0],
            [1999, 1, 1, 0, 0, 0],
            [2006, 1, 1, 0, 0, 0],
            [2009, 1, 1, 0, 0, 0],
            [2012, 7, 1, 0, 0, 0],
            [2015, 7, 1, 0, 0, 0],
            [2017, 1, 1, 0, 0, 0],
        ]
    )

    table_j_days = julian_day(utc_table) - GpsConstants.GPSEPOCHJD
    table_seconds = table_j_days * GpsConstants.DAYSEC
    j_days = julian_day(utc_time) - GpsConstants.GPSEPOCHJD
    time_seconds = j_days * GpsConstants.DAYSEC

    leap_secs = np.zeros(m, dtype=int)
    for i in range(m):
        leap_secs[i] = np.sum(table_seconds <= time_seconds[i])

    return leap_secs
