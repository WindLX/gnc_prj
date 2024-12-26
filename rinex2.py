import re
from enum import Enum
from dataclasses import dataclass

import numpy as np

from utils import date_to_gps, GM, Omegae_dot, F
from model import Vector3

DATA_PATTERN = r"-?\d+\.\d+D[+-]\d+"


def parse_line(line, pattern=DATA_PATTERN) -> list[float]:
    return [float(d.replace("D", "e")) for d in re.findall(pattern, line)]


class SatelliteSystem(Enum):
    GPS = "G"
    GLONASS = "R"
    GeostationarySignalPayload = "S"
    GALILEO = "E"
    BEIDOU = "C"


@dataclass
class RINEXNavigationData:
    svprn: int
    af2: float
    M0: float
    sqrt_a: float
    deltan: float
    ecc: float
    omega: float
    cuc: float
    cus: float
    crc: float
    crs: float
    i0: float
    IDOT: float
    cic: float
    cis: float
    Omega0: float
    Omega_dot: float
    toe: float
    af0: float
    af1: float
    toc: float
    IODE: float
    IODC: float
    weekno: int
    L2flag: float
    svaccur: float
    svhealth: float
    tgd: float
    tom_gps: float
    toe_gps: float
    toc_gps: float

    @staticmethod
    def parse_nav_head(file):
        iono = [0] * 8
        iono_loaded = False
        header_end = False
        a0, a1, SOW, weeknum = None, None, None, None

        while not header_end:
            line = file.readline()
            if "RINEX VERSION / TYPE" in line:
                version = int(line[:9].strip())
            elif (
                "ION ALPHA" in line or "IONOSPHERIC CORR" in line
            ) and not iono_loaded:
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

    @staticmethod
    def read_rinex_nav(
        nav_file: str, time: list
    ) -> tuple[tuple, dict[int, "RINEXNavigationData"]]:
        """
        Reads RINEX navigation file and extracts ephemeris data for given time.

        Args:
            nav_file (str): Path to the RINEX navigation file.
            time (list): List containing the date and time [year, month, day, hour, minute, second].

        Returns:
            tuple: A tuple containing:
                - head (tuple): Parsed header information from the RINEX file.
                - eph (dict[int, RINEXNavigationData]): Dictionary mapping satellite PRN to its corresponding ephemeris data.

        Raises:
            ValueError: If corresponding ephemeris data cannot be found within 2 hours of the required time.
        """
        eph = {i: None for i in range(1, 33)}
        is_found = np.full(32, False)

        with open(nav_file, "r") as file:
            head = RINEXNavigationData.parse_nav_head(file)

            last_search = -86400 * np.ones(32)
            t = 3600 * time[3] + 60 * time[4] + time[5]

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

                af0, af1, af2 = parse_line(lines[0][21:])
                IODE, crs, deltan, M0 = parse_line(lines[1])
                cuc, ecc, cus, sqrt_a = parse_line(lines[2])
                toe, cic, Omega0, cis = parse_line(lines[3])
                i0, crc, omega, Omega_dot = parse_line(lines[4])
                IDOT, code_on_L2, weekno, L2flag = parse_line(lines[5])
                svaccur, svhealth, tgd, IODC = parse_line(lines[6])
                tom = parse_line(lines[7])[0]

                date = [year, month, day, hour, minute, int(second)]
                _, toc, _ = date_to_gps(date)

                current_time = 3600 * hour + 60 * minute + second
                time_diff = abs(current_time - t)
                last_diff = abs(last_search[svprn - 1] - t)

                if time_diff < 7200 and not is_found[svprn - 1]:
                    if last_search[svprn - 1] < t <= current_time:
                        is_found[svprn - 1] = True
                        if last_diff <= time_diff:
                            continue
                    last_search[svprn - 1] = current_time
                    eph.update(
                        {
                            svprn: RINEXNavigationData(
                                svprn,
                                af2,
                                M0,
                                sqrt_a,
                                deltan,
                                ecc,
                                omega,
                                cuc,
                                cus,
                                crc,
                                crs,
                                i0,
                                IDOT,
                                cic,
                                cis,
                                Omega0,
                                Omega_dot,
                                toe,
                                af0,
                                af1,
                                toc,
                                IODE,
                                IODC,
                                weekno,
                                L2flag,
                                svaccur,
                                svhealth,
                                tgd,
                                weekno * 7 * 86400 + tom,
                                weekno * 7 * 86400 + toe,
                                weekno * 7 * 86400 + toc,
                            )
                        }
                    )

                if not lines[0]:
                    if any(value is None for value in eph.values()):
                        raise ValueError(
                            "Cannot find corresponding ephemeris within 2 hours at required time."
                        )
        return head, eph

    @staticmethod
    def rinex_nav_to_sv(
        eph: dict[int, "RINEXNavigationData"], time: list
    ) -> dict[int, tuple[Vector3, float]]:
        """
        Convert RINEX navigation data to satellite vehicle (SV) positions and clock biases.

        Args:
            eph (dict[int, RINEXNavigationData]): A dictionary where the keys are satellite PRNs (Pseudo-Random Numbers)
                and the values are RINEX navigation data objects.
            time (list): A list representing the current GPS time in the format [year, month, day, hour, minute, second].

        Returns:
            dict[int, tuple[Vector3, float]]: A dictionary where the keys are satellite PRNs and the values are tuples
                containing the satellite's ECEF (Earth-Centered, Earth-Fixed) position as a Vector3 object and the clock bias as a float.
        """
        gps_week, gps_sow, gps_dow = date_to_gps(time)
        T = gps_week * 7 * 86400 + gps_sow
        epsilon = 1e-6
        sv_dict = {i: None for i in range(1, 33)}

        for svprn, data in eph.items():
            delta_t = (
                data.af0
                + data.af1 * (T - data.toc_gps)
                + data.af2 * (T - data.toc_gps) ** 2
                - data.tgd
            )
            A = data.sqrt_a**2
            n0 = np.sqrt(GM / A**3)
            tk = T + delta_t - data.toe_gps
            n = n0 + data.deltan
            M = data.M0 + n * tk

            E = M
            E0 = M - epsilon * 2
            while abs(E - E0) > epsilon:
                E0 = E
                E = E0 + (M - E0 + data.ecc * np.sin(E0)) / (1 - data.ecc * np.cos(E0))

            dtr = F * data.ecc * data.sqrt_a * np.sin(E)
            delta_t += dtr
            tk = T + delta_t - data.toe_gps

            v = 2 * np.arctan(np.sqrt((1 + data.ecc) / (1 - data.ecc)) * np.tan(E / 2))
            phi = v + data.omega
            delta_u = data.cus * np.sin(2 * phi) + data.cuc * np.cos(2 * phi)
            delta_r = data.crs * np.sin(2 * phi) + data.crc * np.cos(2 * phi)
            delta_i = data.cis * np.sin(2 * phi) + data.cic * np.cos(2 * phi)

            u = phi + delta_u
            r = A * (1 - data.ecc * np.cos(E)) + delta_r
            i = data.i0 + delta_i + data.IDOT * tk
            x = r * np.cos(u)
            y = r * np.sin(u)

            sv = np.array([x, y, 0])
            Omega_p = (
                data.Omega0 + data.Omega_dot * tk - Omegae_dot * (gps_sow + delta_t)
            )
            r1 = np.array(
                [[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]]
            )
            r3 = np.array(
                [
                    [np.cos(Omega_p), -np.sin(Omega_p), 0],
                    [np.sin(Omega_p), np.cos(Omega_p), 0],
                    [0, 0, 1],
                ]
            )
            sv_ecef = r3 @ r1 @ sv
            sv_ecef = Vector3.from_list(sv_ecef)
            sv_dict.update({svprn: (sv_ecef, float(delta_t))})
        return sv_dict


@dataclass
class RINEXObservationData:
    system: str
    prn: int
    c1: float  # m
    c2: float
    l1: float  # cycle
    l2: float
    p1: float  # m
    p2: float
    s1: float  # db-Hz
    s2: float

    @staticmethod
    def read_observation_file(
        obs_file: str,
    ) -> list[tuple[list, list["RINEXObservationData"]]]:
        """
        Reads a RINEX observation file and extracts observation data at specified time intervals.

        Args:
            obs_file (str): The path to the RINEX observation file.

        Returns:
            list[RINEXObservationData]: A list of RINEXObservationData objects containing the extracted observation data.

        The function reads the observation file line by line, searching for the specified date "22 10 02".
        Once the date is found, it extracts observation data at intervals specified by the step parameter.
        The extracted data includes the time, available satellites (SV), and pseudo-range measurements.
        """
        observations = []
        with open(obs_file, "r") as file:
            line = file.readline()
            header_end = False
            while not header_end:
                line = file.readline()
                if "END OF HEADER" in line:
                    header_end = True

            while True:
                line = file.readline()
                if not line:
                    break

                if re.match(r"^\s\d{2}\s\d{2}\s\d{2}", line):
                    time_str = line[:26].strip()
                    time_parts = time_str.split()
                    year, month, day = (
                        int(time_parts[0]) + 2000,
                        int(time_parts[1]),
                        int(time_parts[2]),
                    )
                    hour, minute, second = (
                        int(time_parts[3]),
                        int(time_parts[4]),
                        int(float(time_parts[5])),
                    )

                    time = [year, month, day, hour, minute, second]

                    next_line = file.readline()
                    satellites_str = line[30:80].strip() + next_line.strip()
                    satellites_count = int(satellites_str[:2])
                    satellites_str = satellites_str[2:]
                    satellites = {
                        satellites_str[i : i + 3]: None
                        for i in range(0, len(satellites_str), 3)
                    }
                    assert len(satellites) == satellites_count

                    element = [None] * 8
                    for svprn, _ in satellites.items():
                        line = file.readline()
                        next_line = file.readline()

                        element[0] = line[:14].strip()
                        element[1] = line[14:30].strip()
                        element[2] = line[30:47].strip()
                        element[3] = line[47:63].strip()
                        element[4] = line[63:].strip()
                        element[5] = next_line[:14].strip()
                        element[6] = next_line[14:30].strip()
                        element[7] = next_line[30:47].strip()
                        element = list(map(lambda e: float(e) if e else None, element))

                        system = SatelliteSystem(svprn[0])
                        prn = int(svprn[1:])
                        data = RINEXObservationData(system, prn, *element)
                        satellites.update({svprn: data})
                    observations.append((time, list(satellites.values())))

        return observations


if __name__ == "__main__":
    t = [2022, 10, 2, 0, 0, 0]
    head, rinex = RINEXNavigationData.read_rinex_nav("brdc2750.22n", t)
    sv = RINEXNavigationData.rinex_nav_to_sv(rinex, t)
    print(rinex)
    for svprn, data in sv.items():
        print(f"{svprn}:\t{data}")

    obs = RINEXObservationData.read_observation_file("bake2750.22o")
    for data in obs:
        print(data)
        print()
