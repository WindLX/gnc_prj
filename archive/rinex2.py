import re
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from model import date2gps, ECEFData, SVData, GM, Omegae_dot


DATA_PATTERN = r"-?\d+\.\d+D[+-]\d+"


def parse_line(line, pattern=DATA_PATTERN) -> list[float]:
    return [float(d.replace("D", "e")) for d in re.findall(pattern, line)]


def parse_head(file):
    iono = [0] * 8
    iono_loaded = False
    header_end = False
    a0, a1, SOW, weeknum = None, None, None, None

    while not header_end:
        line = file.readline()
        if "RINEX VERSION / TYPE" in line:
            version = int(line[:9].strip())
        elif ("ION ALPHA" in line or "IONOSPHERIC CORR" in line) and not iono_loaded:
            data = parse_line(line)
            if len(data) == 4:
                iono[:4] = data
                line = file.readline()
                data = parse_line(line)
                if len(data) == 4:
                    iono[4:] = data
                else:
                    iono = [0] * 8
            iono_loaded = True
        elif "DELTA-UTC: A0,A1,T,W" in line:
            data = parse_line(line)
            if len(data) == 2:
                a0, a1 = data
                SOW = int(line[41:50].strip())
                weeknum = int(line[50:60].strip())
        elif "LEAP SECONDS" in line:
            leap_seconds = int(line[:6].strip())
        elif "END OF HEADER" in line:
            header_end = True

    return version, iono, a0, a1, SOW, weeknum, leap_seconds


@dataclass
class RINEXData:
    svprn: int
    svhealth: float
    ecc: float
    toe: float
    i0: float
    Omega_dot: float
    sqrt_a: float
    Omega: float
    omega: float
    M0: float
    deltan: float
    cuc: float
    cus: float
    crc: float
    crs: float
    idot: float
    cic: float
    cis: float
    af0: float
    af1: float
    af2: float
    weekno: int
    toc: float
    date: datetime


def read_rinex_nav(nav_file: str) -> tuple[tuple, list[RINEXData]]:
    eph = []

    with open(nav_file, "r") as file:
        head = parse_head(file)

        while True:
            lines = [file.readline().strip() for _ in range(8)]
            if not lines[0]:
                break

            start_info = lines[0][:21]
            start_info = start_info.split()
            svprn = int(start_info[0])
            year = int(start_info[1])
            month = int(start_info[2])
            day = int(start_info[3])
            hour = int(start_info[4])
            minute = int(start_info[5])
            second = float(start_info[6])

            af0, af1, af2 = parse_line(lines[0][22:])

            IODE, crs, deltan, M0 = parse_line(lines[1])
            cuc, ecc, cus, roota = parse_line(lines[2])
            toe, cic, Omega0, cis = parse_line(lines[3])
            i0, crc, omega, Omegadot = parse_line(lines[4])
            idot, code_on_L2, weekno, L2flag = parse_line(lines[5])
            svaccur, svhealth, tgd, IODC = parse_line(lines[6])
            tom = parse_line(lines[7])[0]

            date = datetime(year, month, day, hour, minute, int(second))
            toc = date2gps(date).sow

            eph.append(
                RINEXData(
                    svprn=svprn,
                    M0=M0,
                    sqrt_a=roota,
                    deltan=deltan,
                    ecc=ecc,
                    omega=omega,
                    cuc=cuc,
                    cus=cus,
                    crc=crc,
                    crs=crs,
                    i0=i0,
                    idot=idot,
                    cic=cic,
                    cis=cis,
                    Omega=Omega0,
                    Omega_dot=Omegadot,
                    toe=toe,
                    af0=af0,
                    af1=af1,
                    af2=af2,
                    weekno=weekno,
                    svhealth=svhealth,
                    toc=toc,
                    date=date,
                )
            )
    return head, eph


def rinex_to_sv(rinex: RINEXData, t: datetime) -> SVData:
    # t = rinex.toc
    t = date2gps(t).sow

    toe = rinex.toe
    delta_t = rinex.af0 + rinex.af1 * (t - toe) + rinex.af2 * (t - toe) ** 2
    t = t - delta_t

    a = rinex.sqrt_a**2
    n0 = np.sqrt(GM / a**3)
    n = n0 + rinex.deltan
    M = rinex.M0 + n * (t - toe)
    M = np.mod(M, 2 * np.pi)

    E = M
    for _ in range(10):
        E = E - (E - rinex.ecc * np.sin(E) - M) / (1 - rinex.ecc * np.cos(E))

    x = a * np.cos(E) - a * rinex.ecc
    y = a * np.sqrt(1 - rinex.ecc**2) * np.sin(E)

    return SVData(x, y)


def rinex_to_ecef(rinex: RINEXData, t: datetime) -> ECEFData:
    # t = rinex.toc
    t = date2gps(t).sow

    toe = rinex.toe
    delta_t = rinex.af0 + rinex.af1 * (t - toe) + rinex.af2 * (t - toe) ** 2
    t = t - delta_t

    a = rinex.sqrt_a**2
    n0 = np.sqrt(GM / a**3)
    n = n0 + rinex.deltan
    M = rinex.M0 + n * (t - toe)
    M = np.mod(M, 2 * np.pi)

    E = M
    for _ in range(10):
        E = E - (E - rinex.ecc * np.sin(E) - M) / (1 - rinex.ecc * np.cos(E))

    fk = np.arctan2(np.sqrt(1 - rinex.ecc**2) * np.sin(E), np.cos(E) - rinex.ecc)
    phik = fk + rinex.omega
    phik = np.mod(phik, 2 * np.pi)

    uk = phik + rinex.cuc * np.cos(2 * phik) + rinex.cus * np.sin(2 * phik)
    rk = (
        a * (1 - rinex.ecc * np.cos(E))
        + rinex.crc * np.cos(2 * phik)
        + rinex.crs * np.sin(2 * phik)
    )
    i = (
        rinex.i0
        + rinex.idot * (t - toe)
        + rinex.cic * np.cos(2 * phik)
        + rinex.cis * np.sin(2 * phik)
    )

    x_p = np.cos(uk) * rk
    y_p = np.sin(uk) * rk

    Omega_p = rinex.Omega + rinex.Omega_dot * (t - toe) - Omegae_dot * t
    Omega_p = np.mod(Omega_p, 2 * np.pi)

    x_ecef = x_p * np.cos(Omega_p) - y_p * np.cos(i) * np.sin(Omega_p)
    y_ecef = x_p * np.sin(Omega_p) + y_p * np.cos(i) * np.cos(Omega_p)
    z_ecef = y_p * np.sin(i)

    return ECEFData(x_ecef, y_ecef, z_ecef)


if __name__ == "__main__":
    head, rinex = read_rinex_nav("brdc2750.22n")
    date = datetime(2022, 10, 2, 0, 0, 0)
    ecef = rinex_to_ecef(rinex[0], date)
    sv = rinex_to_sv(rinex[0], date)
    print(rinex[0])
    print(sv)
    print(ecef)
