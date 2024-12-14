import numpy as np
from gps_constants import GpsConstants
from leap_seconds import leap_seconds


def fct_to_ymdhms(fctSeconds):
    HOURSEC = 3600
    MINSEC = 60
    monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    m = len(fctSeconds)
    days = np.floor(fctSeconds / GpsConstants.DAYSEC) + 6
    years = np.zeros(m) + 1980
    leap = np.ones(m)

    while np.any(days > (leap + 365)):
        I = np.where(days > (leap + 365))
        days[I] = days[I] - (leap[I] + 365)
        years[I] = years[I] + 1
        leap[I] = np.mod(years[I], 4) == 0

    time = np.zeros((m, 6))
    time[:, 0] = years

    for i in range(m):
        month = 1
        if np.mod(years[i], 4) == 0:
            monthDays[1] = 29
        else:
            monthDays[1] = 28
        while days[i] > monthDays[month - 1]:
            days[i] = days[i] - monthDays[month - 1]
            month = month + 1
        time[i, 1] = month

    time[:, 2] = days

    sinceMidnightSeconds = np.mod(fctSeconds, GpsConstants.DAYSEC)
    time[:, 3] = np.fix(sinceMidnightSeconds / HOURSEC)
    lastHourSeconds = np.mod(sinceMidnightSeconds, HOURSEC)
    time[:, 4] = np.fix(lastHourSeconds / MINSEC)
    time[:, 5] = np.mod(lastHourSeconds, MINSEC)

    return time


def gps_to_utc(gpsTime, fctSeconds=None):
    if fctSeconds is None:
        if gpsTime.shape[1] != 2:
            raise ValueError("gpsTime must have two columns")
        fctSeconds = gpsTime @ np.array([GpsConstants.WEEKSEC, 1])

    fct2100 = np.array([6260, 432000]) @ np.array([GpsConstants.WEEKSEC, 1])
    if np.any(fctSeconds < 0) or np.any(fctSeconds >= fct2100):
        raise ValueError(
            "gpsTime must be in this range: [0,0] <= gpsTime < [6260, 432000]"
        )

    time = fct_to_ymdhms(fctSeconds)
    ls = leap_seconds(time)
    timeMLs = fct_to_ymdhms(fctSeconds - ls)
    ls1 = leap_seconds(timeMLs)

    if np.all(ls1 == ls):
        utcTime = timeMLs
    else:
        utcTime = fct_to_ymdhms(fctSeconds - ls1)

    return utcTime
