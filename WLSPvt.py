import numpy as np

def WlsPvt(prs, gpsEph, xo):
    # ...existing code...
    jWk, jSec, jSv, jPr, jPrSig, jPrr, jPrrSig = 0, 1, 2, 3, 4, 5, 6

    bOk, numVal = checkInputs(prs, gpsEph, xo)
    if not bOk:
        raise ValueError('inputs not right size, or not properly aligned with each other')

    xHat, z, H, svPos = [], [], [], []
    xyz0 = xo[:3]
    bc = xo[3]

    if numVal < 4:
        return xHat, z, svPos, H, None, None

    ttxWeek = prs[:, jWk]
    ttxSeconds = prs[:, jSec] - prs[:, jPr] / GpsConstants.LIGHTSPEED
    dtsv = GpsEph2Dtsv(gpsEph, ttxSeconds)
    ttx = ttxSeconds - dtsv

    svXyzTtx, dtsv, svXyzDot, dtsvDot = GpsEph2Pvt(gpsEph, np.column_stack((ttxWeek, ttx)))
    svXyzTrx = svXyzTtx.copy()

    Wpr = np.diag(1.0 / prs[:, jPrSig])
    Wrr = np.diag(1.0 / prs[:, jPrrSig])

    xHat = np.zeros(4)
    dx = xHat + np.inf
    whileCount, maxWhileCount = 0, 100

    while np.linalg.norm(dx) > GnssThresholds.MAXDELPOSFORNAVM:
        whileCount += 1
        assert whileCount < maxWhileCount, f'while loop did not converge after {whileCount} iterations'
        for i in range(len(gpsEph)):
            dtflight = (prs[i, jPr] - bc) / GpsConstants.LIGHTSPEED + dtsv[i]
            svXyzTrx[i, :] = FlightTimeCorrection(svXyzTtx[i, :], dtflight)

        v = xyz0[:, np.newaxis] - svXyzTrx.T
        range_ = np.sqrt(np.sum(v ** 2, axis=0))
        v = v / range_

        svPos = np.column_stack((prs[:, jSv], svXyzTrx, dtsv))

        prHat = range_ + bc - GpsConstants.LIGHTSPEED * dtsv
        zPr = prs[:, jPr] - prHat
        H = np.column_stack((v.T, np.ones(numVal)))

        dx = np.linalg.pinv(Wpr @ H) @ (Wpr @ zPr)

        xHat += dx
        xyz0 += dx[:3]
        bc += dx[3]

        zPr -= H @ dx

    rrMps = np.zeros(numVal)
    for i in range(numVal):
        rrMps[i] = -np.dot(svXyzDot[i, :], v[:, i])

    prrHat = rrMps + xo[7] - GpsConstants.LIGHTSPEED * dtsvDot
    zPrr = prs[:, jPrr] - prrHat
    vHat = np.linalg.pinv(Wrr @ H) @ (Wrr @ zPrr)
    xHat = np.concatenate((xHat, vHat))

    z = np.concatenate((zPr, zPrr))

    return xHat, z, svPos, H, Wpr, Wrr

def checkInputs(prs, gpsEph, xo):
    # ...existing code...
    jWk, jSec, jSv, jPr, jPrSig, jPrr, jPrrSig = 0, 1, 2, 3, 4, 5, 6

    bOk = False
    numVal = prs.shape[0]
    if (np.max(prs[:, jSec]) - np.min(prs[:, jSec])) > np.finfo(float).eps:
        return bOk, numVal
    elif len(gpsEph) != numVal:
        return bOk, numVal
    elif not np.array_equal(prs[:, jSv], np.array([eph.PRN for eph in gpsEph])):
        return bOk, numVal
    elif xo.shape != (8,):
        return bOk, numVal
    elif prs.shape[1] != 7:
        return bOk, numVal
    else:
        bOk = True

    return bOk, numVal

# ...existing code...
