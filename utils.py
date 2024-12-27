import numpy as np
from datetime import datetime

from model import Vector3

a = 6378137.0
f = 1 / 298.257223563
e2 = 2 * f - f * f
GM = 3.986005e14
Omegae_dot = 7.2921151467e-5
F = -4.442807633e-10
c = 299792458

# now = [121.441468, 31.025953]
# true = [121.446156, 31.024026]
# bias = [true[0] - now[0], true[1] - now[1]]
bias = [0.004688, -0.001927]


def lla_to_ecef(lla: Vector3) -> Vector3:
    """
    lla_to_ecef Convert latitude, longitude, altitude to ECEF coordinates.

    Parameters:
    lla : Vector3
        Latitude, longitude in degrees and altitude in meters [lat, lon, alt].

    Returns:
    ecef : Vector3
        ECEF coordinates [x, y, z].
    """
    phi = np.radians(lla[0])
    lambda_ = np.radians(lla[1])
    N = a / np.sqrt(1 - e2 * np.sin(phi) * np.sin(phi))
    x = (N + lla[2]) * np.cos(phi) * np.cos(lambda_)
    y = (N + lla[2]) * np.cos(phi) * np.sin(lambda_)
    z = ((1 - e2) * N + lla[2]) * np.sin(phi)
    ecef = Vector3(x, y, z)
    return ecef


def ecef_to_lla(ecef: Vector3) -> Vector3:
    """
    ecef_to_lla Convert ECEF coordinates to latitude, longitude, altitude.

    Parameters:
    ecef : Vector3
        ECEF coordinates [x, y, z].

    Returns:
    lla : Vector3
        Latitude, longitude in degrees and altitude in meters [lat, lon, alt].
    """
    eps = 1e-6
    lla = Vector3.zero()

    if ecef[0] >= 0:
        lla[1] = np.arctan(ecef[1] / ecef[0])
    else:
        if ecef[1] >= 0:
            lla[1] = np.pi + np.arctan(ecef[1] / ecef[0])
        else:
            lla[1] = -np.pi + np.arctan(ecef[1] / ecef[0])
    lla[0] = np.arctan(
        ecef[2] / np.sqrt(ecef[0] ** 2 + ecef[1] ** 2) * (1 + e2 / (1 - e2))
    )
    tmp = lla[0] + 10 * eps

    while abs(lla[0] - tmp) > eps:
        tmp = lla[0]
        N = a / np.sqrt(1 - e2 * np.sin(lla[0]) ** 2)
        lla[0] = np.arctan(
            ecef[2]
            / np.sqrt(ecef[0] ** 2 + ecef[1] ** 2)
            * (1 + e2 * N * np.sin(lla[0]) / ecef[2])
        )

    lla[2] = (
        np.sqrt(ecef[0] ** 2 + ecef[1] ** 2) * np.cos(lla[0])
        + ecef[2] * np.sin(lla[0])
        - a * np.sqrt(1 - e2 * np.sin(lla[0]) ** 2)
    )
    lla[0] = np.degrees(lla[0])
    lla[1] = np.degrees(lla[1])

    lla = Vector3.from_list(lla)
    return lla


def ecef_to_enu(ecef: Vector3, target_point: Vector3) -> tuple[Vector3, float, float]:
    """
    ecef_to_enu Convert ECEF coordinates to ENU coordinates.

    Parameters:
    ecef : Vector3
        Target coordinates in ECEF [x, y, z].
    point : Vector3
        ENU's origin coordinates in LLA [lat, lon, alt].

    Returns:
    enu : Vector3
        ENU coordinates [e, n, u].
    az : float
        Azimuth angle in radians.
    el : float
        Elevation angle in radians.
    """
    phi = target_point[0] * np.pi / 180
    lambda_ = target_point[1] * np.pi / 180
    point_ecef = lla_to_ecef(target_point)
    ecef_diff = (ecef - point_ecef).numpy()
    A = np.array(
        [
            [-np.sin(lambda_), np.cos(lambda_), 0],
            [
                -np.cos(lambda_) * np.sin(phi),
                -np.sin(lambda_) * np.sin(phi),
                np.cos(phi),
            ],
            [np.cos(lambda_) * np.cos(phi), np.sin(lambda_) * np.cos(phi), np.sin(phi)],
        ]
    )
    enu = A @ ecef_diff

    if enu[1] >= 0:
        az = np.arctan(enu[0] / enu[1])
    else:
        az = np.pi + np.arctan(enu[0] / enu[1])
    el = np.arcsin(enu[2] / np.linalg.norm(enu))

    enu = Vector3.from_list(enu)
    return enu, az, el


def enu_to_ecef(enu: Vector3, target_point: Vector3) -> Vector3:
    """
    enu_to_ecef Convert ENU coordinates to ECEF coordinates.

    Parameters:
    enu : Vector3
        Target coordinates in ENU [e, n, u].
    point : Vector3
        ENU's origin coordinates in LLA [lat, lon, alt].

    Returns:
    ecef : Vector3
        ECEF coordinates [x, y, z].
    """
    phi = target_point[0] * np.pi / 180
    lambda_ = target_point[1] * np.pi / 180
    point_ecef = lla_to_ecef(target_point)
    A = np.array(
        [
            [-np.sin(lambda_), np.cos(lambda_), 0],
            [
                -np.cos(lambda_) * np.sin(phi),
                -np.sin(lambda_) * np.sin(phi),
                np.cos(phi),
            ],
            [np.cos(lambda_) * np.cos(phi), np.sin(lambda_) * np.cos(phi), np.sin(phi)],
        ]
    )
    ecef = point_ecef.numpy() + A.T @ enu.numpy()
    ecef = Vector3.from_list(ecef)
    return ecef


def date_to_gps(date: list) -> tuple[int, int, int]:
    date_d = datetime(*date[:3])
    gps_start_date = datetime(1980, 1, 6)
    delta = date_d - gps_start_date
    deltat = delta.total_seconds() / 86400
    gps_week = deltat // 7
    gps_dow = deltat - gps_week * 7
    gps_sow = (deltat - gps_week * 7) * 86400
    gps_sow += date[3] * 3600 + date[4] * 60 + date[5]

    return gps_week, gps_sow, gps_dow


if __name__ == "__main__":
    ground_truth = 1.0e6 * np.array([-0.2898, -2.7565, 5.7252])
    local_pos = ecef_to_lla(ground_truth)
    print(f"Local position: {local_pos}")
