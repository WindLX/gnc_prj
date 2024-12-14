import numpy as np
from gps_eph_to_xyz import gps_eph_to_xyz


# Assuming GpsEph2Xyz is already implemented as we discussed before.
# Here we implement GpsEph2Pvt, which computes the satellite position, velocity, and clock error rate.


def gps_eph_to_pvt(gps_eph, gps_time):
    """
    Calculate satellite coordinates, clock bias, and satellite velocity.

    Inputs:
        gps_eph: List of GPS ephemeris objects (each object should have the fields like TGD, Toc, etc.)
        gps_time: A numpy array or list with shape (n, 2), where the first column is gpsWeek and the second column is ttxSec.

    Outputs:
        xM: Matrix of satellite coordinates in ECEF frame (meters)
        dtsvS: Vector of satellite clock bias (seconds)
        vMps: Matrix of satellite velocities in ECEF frame (m/s)
        dtsvSDot: Vector of satellite clock error rate (seconds/second)
    """
    vMps = None
    dtsvSDot = None

    # Get satellite positions and clock bias at gpsTime (t=0)
    xM, dtsvS = gps_eph_to_xyz(gps_eph, gps_time)
    if xM is None:
        return None, None, None, None

    # Compute velocity from delta position and dtsvS at t+0.5 - t-0.5
    # Time + 0.5 seconds
    t1 = np.copy(gps_time)
    t1[:, 1] += 0.5  # Add 0.5 to the second column (time in seconds)
    xPlus, dtsvPlus = gps_eph_to_xyz(gps_eph, t1)

    # Time - 0.5 seconds
    t1[:, 1] -= 1.0  # Subtract 1.0 to make the second column (time - 0.5)
    xMinus, dtsvMinus = gps_eph_to_xyz(gps_eph, t1)

    # Compute satellite velocity (difference of positions at t+0.5 and t-0.5)
    vMps = xPlus - xMinus
    # Compute satellite clock error rate (difference of dtsvS at t+0.5 and t-0.5)
    dtsvSDot = dtsvPlus - dtsvMinus

    return xM, dtsvS, vMps, dtsvSDot


# Assuming gps_eph_to_xyz is the previously implemented function to get satellite positions and clock bias.
# You can replace this with the actual function call.
