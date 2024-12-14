import numpy as np


def julian_day(utcTime):
    """
    Convert UTC time to Julian Day.

    Parameters:
    utcTime (numpy.ndarray): [mx6] matrix [year, month, day, hours, minutes, seconds]

    Returns:
    numpy.ndarray: totalDays in Julian Days [mx1] vector (real number of days)
    """
    if utcTime.shape[1] != 6:
        raise ValueError("utcTime must have 6 columns")

    y = utcTime[:, 0]
    m = utcTime[:, 1]
    d = utcTime[:, 2]
    h = utcTime[:, 3] + utcTime[:, 4] / 60 + utcTime[:, 5] / 3600

    if np.any(y < 1901) or np.any(y > 2099):
        raise ValueError("utcTime[:, 0] not in allowed range: 1900 < year < 2100")

    i2 = m <= 2
    m[i2] += 12
    y[i2] -= 1

    jDays = (
        np.floor(365.25 * y) + np.floor(30.6001 * (m + 1)) - 15 + 1720996.5 + d + h / 24
    )

    return jDays
