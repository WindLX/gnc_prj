import numpy as np
from check_gps_eph_inputs import check_gps_eph_inputs
from gps_constants import GpsConstants

# Assuming the constants like GpsConstants.WEEKSEC, GpsConstants.mu, and GpsConstants.FREL are defined elsewhere.
# Also assuming the Kepler solver function (Kepler) is implemented.


def gps_eph_to_xyz(gps_eph, gps_time):
    """
    Calculate satellite coordinates in ECEF frame at the time of transmission.

    Inputs:
        gps_eph: List of GPS ephemeris structures (each structure should have fields like TGD, Toc, etc.)
        gps_time: A numpy array or list with shape (n, 2), where the first column is gpsWeek and the second column is ttxSec.

    Outputs:
        xyzM: Matrix of satellite coordinates in ECEF frame (meters)
        dtsvS: Vector of satellite clock bias (seconds)
    """
    # Initialize output variables
    xyzM = []
    dtsvS = []

    # Check inputs and preprocess (CheckGpsEphInputs equivalent in Python)
    bOk, gps_eph, gps_week, ttx_sec = check_gps_eph_inputs(gps_eph, gps_time)
    if not bOk:
        return None, None

    p = len(gps_eph)

    # Set fitIntervalSeconds (adjust fitIntervalHours to seconds)
    fit_interval_hours = np.array([eph["Fit_interval"] for eph in gps_eph])
    fit_interval_hours[fit_interval_hours == 0] = 2  # Adjust for zeros
    fit_interval_seconds = fit_interval_hours * 3600  # Convert hours to seconds

    # Extract variables from gps_eph (mapping ephemeris fields to column vectors)
    TGD = np.array([eph["TGD"] for eph in gps_eph])
    Toc = np.array([eph["Toc"] for eph in gps_eph])
    af2 = np.array([eph["af2"] for eph in gps_eph])
    af1 = np.array([eph["af1"] for eph in gps_eph])
    af0 = np.array([eph["af0"] for eph in gps_eph])
    Crs = np.array([eph["Crs"] for eph in gps_eph])
    Delta_n = np.array([eph["Delta_n"] for eph in gps_eph])
    M0 = np.array([eph["M0"] for eph in gps_eph])
    Cuc = np.array([eph["Cuc"] for eph in gps_eph])
    e = np.array([eph["e"] for eph in gps_eph])
    Cus = np.array([eph["Cus"] for eph in gps_eph])
    Asqrt = np.array([eph["Asqrt"] for eph in gps_eph])
    Toe = np.array([eph["Toe"] for eph in gps_eph])
    Cic = np.array([eph["Cic"] for eph in gps_eph])
    OMEGA = np.array([eph["OMEGA"] for eph in gps_eph])
    Cis = np.array([eph["Cis"] for eph in gps_eph])
    i0 = np.array([eph["i0"] for eph in gps_eph])
    Crc = np.array([eph["Crc"] for eph in gps_eph])
    omega = np.array([eph["omega"] for eph in gps_eph])
    OMEGA_DOT = np.array([eph["OMEGA_DOT"] for eph in gps_eph])
    IDOT = np.array([eph["IDOT"] for eph in gps_eph])
    eph_gps_week = np.array([eph["GPS_Week"] for eph in gps_eph])

    # Calculate dependent variables
    tk = (gps_week - eph_gps_week) * GpsConstants.WEEKSEC + (ttx_sec - Toe)

    I = np.where(np.abs(tk) > fit_interval_seconds)[0]
    if len(I) > 0:
        num_times = len(I)
        print(f"WARNING: {num_times} times outside fit interval.")

    A = Asqrt**2  # Semi-major axis of orbit
    n0 = np.sqrt(GpsConstants.mu / (A**3))  # Computed mean motion (rad/sec)
    n = n0 + Delta_n  # Corrected mean motion
    h = np.sqrt(A * (1 - e**2) * GpsConstants.mu)
    Mk = M0 + n * tk  # Mean anomaly
    Ek = kepler(Mk, e)  # Solve Kepler's equation for eccentric anomaly

    # Calculate satellite clock bias
    dt = (gps_week - eph_gps_week) * GpsConstants.WEEKSEC + (ttx_sec - Toc)
    sin_Ek = np.sin(Ek)
    cos_Ek = np.cos(Ek)
    dtsvS = (
        af0 + af1 * dt + af2 * (dt**2) + GpsConstants.FREL * e * Asqrt * sin_Ek - TGD
    )

    # True anomaly
    vk = np.atan2(
        np.sqrt(1 - e**2) * sin_Ek / (1 - e * cos_Ek), (cos_Ek - e) / (1 - e * cos_Ek)
    )
    Phik = vk + omega  # Argument of latitude

    sin_2Phik = np.sin(2 * Phik)
    cos_2Phik = np.cos(2 * Phik)

    # Second harmonic perturbations
    duk = Cus * sin_2Phik + Cuc * cos_2Phik  # Argument of latitude correction
    drk = Crc * cos_2Phik + Crs * sin_2Phik  # Radius correction
    dik = Cic * cos_2Phik + Cis * sin_2Phik  # Inclination correction

    # Corrected arguments and radius
    uk = Phik + duk
    rk = A * ((1 - e**2) / (1 + e * np.cos(vk))) + drk
    ik = i0 + IDOT * tk + dik  # Corrected inclination

    # Calculate ECEF position
    sin_uk = np.sin(uk)
    cos_uk = np.cos(uk)
    xkp = rk * cos_uk  # Position in orbital plane
    ykp = rk * sin_uk  # Position in orbital plane

    # Corrected longitude of ascending node
    Wk = OMEGA + (OMEGA_DOT - GpsConstants.WE) * tk - GpsConstants.WE * Toe

    # For dtflight (time correction), see FlightTimeCorrection.m
    sin_Wk = np.sin(Wk)
    cos_Wk = np.cos(Wk)

    xyzM = np.zeros((p, 3))
    sin_ik = np.sin(ik)
    cos_ik = np.cos(ik)

    # ECEF coordinates calculation
    xyzM[:, 0] = xkp * cos_Wk - ykp * cos_ik * sin_Wk
    xyzM[:, 1] = xkp * sin_Wk + ykp * cos_ik * cos_Wk
    xyzM[:, 2] = ykp * sin_ik

    return xyzM, dtsvS


# Placeholder for Kepler's equation solver
def kepler(M, e):
    """
    Solve Kepler's equation for eccentric anomaly using an iterative method.

    M: Mean anomaly
    e: Eccentricity

    Returns:
        E: Eccentric anomaly
    """
    E = M  # Initial guess: Eccentric anomaly is the mean anomaly at first
    tolerance = 1e-6
    while True:
        delta = E - e * np.sin(E) - M
        if np.abs(delta) < tolerance:
            break
        E = E - delta / (1 - e * np.cos(E))  # Newton's method
    return E


# Assume GpsConstants, check_gps_eph_inputs, and other required constants are defined elsewhere
