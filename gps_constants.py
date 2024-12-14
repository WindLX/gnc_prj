class GpsConstants:
    """
    GPS constants, defined by WGS84 and IS-GPS-200, and derived from those

    Author: Frank van Diggelen
    Open Source code for processing Android GNSS Measurements
    """

    # Listed alphabetically
    EARTHECCEN2 = 6.69437999014e-3  # WGS 84 (Earth eccentricity)^2 (m^2)
    EARTHMEANRADIUS = 6371009  # Mean R of ellipsoid(m) IU Gedosey& Geophysics
    EARTHSEMIMAJOR = 6378137  # WGS 84 Earth semi-major axis (m)
    EPHVALIDSECONDS = 7200  # +- 2 hours ephemeris validity
    DAYSEC = 86400  # Number of seconds in a day
    FREL = -4.442807633e-10  # Clock relativity parameter, (s/m^1/2)
    GPSEPOCHJD = 2444244.5  # GPS Epoch in Julian Days
    HORIZDEG = 5  # Angle above horizon at which GPS models break down
    LIGHTSPEED = 2.99792458e8  # WGS-84 Speed of light in a vacuum (m/s)
    # Mean time of flight between closest GPS sat (~66 ms) & furthest (~84 ms):
    MEANTFLIGHTSECONDS = 75e-3
    mu = 3.986005e14  # WGS-84 Universal gravitational parameter (m^3/sec^2)
    WE = 7.2921151467e-5  # WGS 84 value of earth's rotation rate (rad/s)
    WEEKSEC = 604800  # Number of seconds in a week
