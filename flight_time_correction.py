import numpy as np


def flight_time_correction(xE, dTflightSeconds):
    """
    Compute rotated satellite ECEF coordinates caused by Earth
    rotation during signal flight time.

    Inputs:
        xE         - satellite ECEF position at time of transmission
        dTflight   - signal flight time (seconds)

    Outputs:
        xERot     - rotated satellite position vector (ECEF at trx)

    Reference:
    IS GPS 200, 20.3.3.4.3.3.2 Earth-Centered, Inertial (ECI) Coordinate System
    """

    # Rotation angle (radians):
    WE = 7.2921151467e-5  # Earth's rotation rate in rad/s
    theta = WE * dTflightSeconds

    # Apply rotation from IS GPS 200-E, 20.3.3.4.3.3.2
    # Note: IS GPS 200-E shows the rotation from ecef to eci
    # so our rotation R3, is in the opposite direction:
    R3 = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    xERot = np.dot(R3, xE)

    return xERot
