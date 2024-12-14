import numpy as np


# Constants (make sure to define these constants)
class GpsConstants:
    WEEKSEC = 604800  # Number of seconds in a week
    mu = 3.986004418e14  # Earth's gravitational constant (m^3/s^2)
    FREL = (
        1.0  # Frequency-related constant (This may need adjustment based on the model)
    )


def gps_eph_to_dtsv(gps_eph, t_s):
    """
    Calculate satellite clock bias (dtsvS).

    Inputs:
        gps_eph: list of GPS ephemeris objects (each object should have the fields TGD, Toc, etc.)
        t_s: numpy array or list of GPS time of week (seconds) at time of transmission.

    Outputs:
        dtsvS: satellite clock bias (seconds).
    """
    # Ensure t_s is a row vector
    t_s = np.array(t_s).flatten()
    pt = len(t_s)
    p = len(gps_eph)

    # Check if gps_eph and t_s are compatible
    if p > 1 and pt != p:
        raise ValueError("If gps_eph is a vector, t_s must have the same length.")

    # Extract the necessary variables from gps_eph
    TGD = np.array([eph.TGD for eph in gps_eph])
    Toc = np.array([eph.Toc for eph in gps_eph])
    af2 = np.array([eph.af2 for eph in gps_eph])
    af1 = np.array([eph.af1 for eph in gps_eph])
    af0 = np.array([eph.af0 for eph in gps_eph])
    Delta_n = np.array([eph.Delta_n for eph in gps_eph])
    M0 = np.array([eph.M0 for eph in gps_eph])
    e = np.array([eph.e for eph in gps_eph])
    Asqrt = np.array([eph.Asqrt for eph in gps_eph])
    Toe = np.array([eph.Toe for eph in gps_eph])

    # Calculate dependent variables
    tk = t_s - Toe  # Time since time of applicability
    tk[tk > 302400.0] -= GpsConstants.WEEKSEC
    tk[tk < -302400.0] += GpsConstants.WEEKSEC

    A = Asqrt**2  # Semi-major axis of orbit
    n0 = np.sqrt(GpsConstants.mu / (A**3))  # Computed mean motion (rad/sec)
    n = n0 + Delta_n  # Corrected mean motion
    Mk = M0 + n * tk  # Mean anomaly
    Ek = kepler(Mk, e)  # Solve Kepler's equation for eccentric anomaly

    # Calculate satellite clock bias (from ICD-GPS-200)
    dt = t_s - Toc
    dt[dt > 302400.0] -= GpsConstants.WEEKSEC
    dt[dt < -302400.0] += GpsConstants.WEEKSEC

    dtsvS = (
        af0
        + af1 * dt
        + af2 * (dt**2)
        + GpsConstants.FREL * e * Asqrt * np.sin(Ek)
        - TGD
    )

    return dtsvS


def kepler(M, e):
    """
    Solve Kepler's equation for eccentric anomaly.

    Inputs:
        M: Mean anomaly (radians)
        e: Eccentricity

    Outputs:
        E: Eccentric anomaly (radians)
    """
    E = M  # Initial guess (E = M for eccentric anomaly)
    for _ in range(10):  # Iterate to solve Kepler's equation
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E
