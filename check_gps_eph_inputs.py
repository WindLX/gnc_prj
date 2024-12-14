def check_gps_eph_inputs(gps_eph, gps_time):
    """
    Check the inputs for GpsEph2Pvt, GpsEph2Xyz, GpsEph2Dtsv

    Author: Frank van Diggelen
    Open Source code for processing Android GNSS Measurements

    Parameters:
    gps_eph (dict): GPS ephemeris data structure
    gps_time (list): List of [gpsWeek, gpsSec] pairs

    Returns:
    tuple: (bOk, gps_eph, gps_week, ttx_sec)
    """
    b_ok = False

    if not isinstance(gps_eph, dict):
        raise ValueError("gpsEph input must be a structure, as defined by ReadRinexNav")

    p = len(gps_eph)

    # Check that gpsTime is a px2 vector
    if len(gps_time) != p or any(len(time) != 2 for time in gps_time):
        raise ValueError(
            "gpsTime must be px2 [gpsWeek, gpsSec], where p = length(gpsEph)"
        )

    gps_week = [time[0] for time in gps_time]
    ttx_sec = [time[1] for time in gps_time]

    b_ok = True

    return b_ok, gps_eph, gps_week, ttx_sec
