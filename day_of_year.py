import numpy as np

from julian_day import julian_day


def day_of_year(utc_time):
    if len(utc_time) != 6:
        raise ValueError("utcTime must be 1x6 for day_of_year function")

    j_day = julian_day([utc_time[0], utc_time[1], utc_time[2], 0, 0, 0])
    j_day_jan1 = julian_day([utc_time[0], 1, 1, 0, 0, 0])
    day_number = j_day - j_day_jan1 + 1

    return day_number
