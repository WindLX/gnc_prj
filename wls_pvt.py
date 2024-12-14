import numpy as np
from gps_eph_to_dtsv import gps_eph_to_dtsv
from gps_eph_to_pvt import gps_eph_to_pvt
from flight_time_correction import flight_time_correction

# Constants
LIGHTSPEED = 299792458  # Speed of light in meters per second


def wls_pvt(prs, gps_eph, xo):
    """
    Weighted Least Squares PVT solution.

    Inputs:
        prs: numpy array of shape (n, 7), raw pseudoranges, and pr rates
        gps_eph: list of GPS ephemeris objects
        xo: initial state, numpy array of shape (8,)

    Outputs:
        xHat: state update estimate
        z: a-posteriori residuals (measured - calculated)
        svPos: matrix of calculated satellite positions and clock error
        H: observation matrix corresponding to svs
        Wpr: Weight matrix for pseudorange
        Wrr: Weight matrix for pseudorange rate
    """
    # Unpack prs (columns)
    jWk, jSec, jSv, jPr, jPrSig, jPrr, jPrrSig = 0, 1, 2, 3, 4, 5, 6

    b_ok, num_val = check_inputs(prs, gps_eph, xo)
    if not b_ok:
        raise ValueError(
            "Inputs not right size or not properly aligned with each other."
        )

    x_hat = np.zeros(4)
    z = []
    H = []
    sv_pos = []
    xyz0 = xo[:3]
    bc = xo[3]

    if num_val < 4:
        return x_hat, z, sv_pos, H, None, None

    ttx_week = prs[:, jWk]
    ttx_seconds = prs[:, jSec] - prs[:, jPr] / LIGHTSPEED

    # GPS time correction for satellite clock error
    dtsv = gps_eph_to_dtsv(gps_eph, ttx_seconds)
    ttx = ttx_seconds - dtsv

    # Calculate satellite position at ttx
    sv_xyz_ttx, dtsv, sv_xyz_dot, dtsv_dot = gps_eph_to_pvt(
        gps_eph, np.column_stack((ttx_week, ttx))
    )
    sv_xyz_trx = sv_xyz_ttx  # Initialize svXyz at time of reception

    # Compute weights
    Wpr = np.diag(1.0 / prs[:, jPrSig])
    Wrr = np.diag(1.0 / prs[:, jPrrSig])

    # Iterative solution
    dx = np.inf * np.ones(4)
    while_count = 0
    max_while_count = 100
    while np.linalg.norm(dx) > 1e-6:  # Max position update threshold
        while_count += 1
        if while_count >= max_while_count:
            raise ValueError(
                f"While loop did not converge after {while_count} iterations."
            )

        for i in range(len(gps_eph)):
            # Calculate flight time correction
            dt_flight = (prs[i, jPr] - bc) / LIGHTSPEED + dtsv[i]
            sv_xyz_trx[i, :] = flight_time_correction(sv_xyz_ttx[i, :], dt_flight)

        # Calculate line of sight vectors and ranges from satellite to xo
        v = np.expand_dims(xyz0, axis=0) - sv_xyz_trx  # Vector from sv to xo
        range_ = np.sqrt(np.sum(v**2, axis=1))
        v = v / range_[:, np.newaxis]  # Line of sight unit vectors from sv to xo

        sv_pos = np.column_stack((prs[:, jSv], sv_xyz_trx, dtsv))

        # Calculate the a-priori range residual
        pr_hat = range_ + bc - LIGHTSPEED * dtsv
        z_pr = prs[:, jPr] - pr_hat
        H = np.column_stack((v, np.ones(num_val)))

        # Solve for position update
        dx = np.linalg.pinv(Wpr @ H) @ Wpr @ z_pr

        # Update state
        x_hat += dx
        xyz0 += dx[:3]
        bc += dx[3]

        # Update a-posteriori range residual
        z_pr = z_pr - H @ dx

    # Compute velocities
    rr_mps = np.zeros(num_val)
    for i in range(num_val):
        rr_mps[i] = -np.dot(sv_xyz_dot[i, :], v[i, :])

    prr_hat = rr_mps + xo[7] - LIGHTSPEED * dtsv_dot
    z_prr = prs[:, jPrr] - prr_hat

    # Solve for velocity update
    v_hat = np.linalg.pinv(Wrr @ H) @ Wrr @ z_prr
    x_hat = np.hstack((x_hat, v_hat))

    z = np.hstack((z_pr, z_prr))

    return x_hat, z, sv_pos, H, Wpr, Wrr


def check_inputs(prs, gps_eph, xo):
    """
    Check if inputs are valid.
    """
    jWk, jSec, jSv, jPr, jPrSig, jPrr, jPrrSig = 0, 1, 2, 3, 4, 5, 6
    b_ok = False
    num_val = prs.shape[0]

    if (np.max(prs[:, jSec]) - np.min(prs[:, jSec])) > np.finfo(float).eps:
        return b_ok, num_val
    if len(gps_eph) != num_val:
        return b_ok, num_val
    if np.any(prs[:, jSv] != np.array([eph.PRN for eph in gps_eph])):
        return b_ok, num_val
    if xo.shape != (8,):
        return b_ok, num_val
    if prs.shape[1] != 7:
        return b_ok, num_val

    b_ok = True
    return b_ok, num_val


# Helper function definitions (e.g., gps_eph_to_dtsv, gps_eph_to_pvt, flight_time_correction) would go here.
