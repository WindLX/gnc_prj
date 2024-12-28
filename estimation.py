from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from model import Vector3
from kml import KML
from rinex4 import RINEXNavigationData, RINEXObservationData, SatelliteSystem
from utils import ecef_to_lla, ecef_to_enu, lla_to_ecef, Omegae_dot, c


class PositionEstimation:
    def __init__(
        self,
        obs_file: str,
        track_file: str,
        epsilon: float = 1e-8,
        max_iterations: int = 1000,
        threshold: float = 0,
        step: int = 1,
        log_dir: str = "log",
        figure_dir: str = "figure",
    ):
        self.obs_file = obs_file
        self.track = KML(track_file)
        self.track.parse()
        self.estimate_epsilon = epsilon
        self.max_iterations = max_iterations
        self.elevation_threshold = threshold
        self.step = step
        self.log_dir = log_dir
        self.figure_dir = figure_dir
        self.estimate_lla = None
        self.inon_data = None

    def load_observation_data(self):
        obs = RINEXObservationData.read_observation_file(self.obs_file)
        return obs[:: self.step]

    def get_truth_position(self, time: list) -> Vector3:
        return self.track.interpolate_at(datetime(*time))

    def extract_satellite_info(
        self,
        time: list[float],
        observation: list[RINEXObservationData],
        nav_file: str,
        target_system: SatelliteSystem = SatelliteSystem.GPS,
    ) -> tuple[list[dict], Vector3]:
        info = []

        target_satellites = [
            data for data in observation if data.system == target_system
        ]

        local_pos = self.get_truth_position(time)
        _head, rinex_nav = RINEXNavigationData.read_rinex_nav(nav_file, time)
        self.inon_data = _head[1]
        # print(f"Iono: {self.Inon_data}")
        sv_info = RINEXNavigationData.rinex_nav_to_sv(rinex_nav, time)

        for data in target_satellites:
            prn = data.prn
            assert prn in sv_info.keys()
            if sv_info[prn] is None:
                continue
            ecef, delta_t = sv_info[prn]
            enu, az, el = ecef_to_enu(ecef, local_pos)
            if (
                self.elevation_threshold
                < np.degrees(el)
                < 90 - self.elevation_threshold
            ):
                info.append(
                    {
                        "system": target_system,
                        "prn": prn,
                        "azimuth": az,
                        "elevation": el,
                        "enu": enu,
                        "ecef": ecef,
                        "delta_t": delta_t,
                        "obs_data": data,
                    }
                )
        return info, local_pos

    def calculate_iono_delay(
        self, time: list[float], satellite_info: list[dict]
    ) -> float:
        length = len(satellite_info)
        dltR = np.zeros(length)
        for idx, data in enumerate(satellite_info):
            az = data["azimuth"]
            el = data["elevation"]
            # print(f"PRN: {data['prn']}, Azimuth: {az}, Elevation: {el}")
            el_ = np.clip(el, np.deg2rad(5), np.deg2rad(90))
            latitude = np.deg2rad(self.estimate_lla[0]) + (
                0.0137 * el_ - 0.11
            ) / np.cos(el)
            longitude = np.deg2rad(self.estimate_lla[1]) + latitude * np.sin(
                az
            ) / np.cos(latitude)
            latitude = latitude + (0.064 * np.cos(longitude - 1.617))
            # print(Latitude)

            localtime = datetime(*time) - datetime(2024, 12, 27, 16, 0, 0)
            localtime = localtime.total_seconds() % 86400
            # print(Localtime)

            A = (
                self.inon_data[0]
                + self.inon_data[1] * latitude
                + self.inon_data[2] * latitude**2
                + self.inon_data[3] * latitude**3
            )
            P = (
                self.inon_data[4]
                + self.inon_data[5] * latitude
                + self.inon_data[6] * latitude**2
                + self.inon_data[7] * latitude**3
            )
            if A < 0:
                A = 0

            x = 2 * np.pi * (localtime - 50400) / P

            Iion = 5e-9
            if x >= -1.57 and x <= 1.57:
                dltR[idx] = Iion + A * np.cos(x)
            else:
                dltR[idx] = Iion
        dltR = dltR * c
        # print(f"Iono: {dltR}, A: {A}, P: {P}, x: {x}, Localtime: {Localtime}")
        return dltR

    def estimate_position(
        self,
        time: list[float],
        truth_lla: Vector3,
        estimation_data: list[dict],
        data_to_use: str = "c1",
    ) -> tuple[np.ndarray, dict]:
        length = len(estimation_data)
        if length < 4:
            raise ValueError("Insufficient Visible Satellite Counts")
        else:
            data = np.zeros((length, 5))
            for idx, d in enumerate(estimation_data):
                data[idx] = [
                    *d["ecef"].list(),
                    d["delta_t"],
                    getattr(d["obs_data"], data_to_use),
                ]
            G = np.zeros((length, 4))
            x = np.zeros(4)
            delta_x = np.ones(3)
            delta_rho = np.zeros(length)

            tz = 77.6e-6 * (101725 / 294.15) * 43000 / 5
            td = 0.373 * (2179 / 294.15**2) * 12000 / 5
            tropo = (tz + td) * 1.001 / np.sqrt(0.002001 + np.sin(d["elevation"]) ** 2)
            # print(f"Tropo: {tropo}")

            # r = pseudo + c * delta_t
            r = data[:, 4] + c * data[:, 3] - tropo * 0.01
            if self.estimate_lla:
                dR = self.calculate_iono_delay(time, estimation_data)
                r = r - dR
            # print(f"r: {r}, dR: {dR}")

            for i in range(length):
                Omega_tau = 1 * Omegae_dot * r[i] / c
                R_sagnac = np.array(
                    [
                        [np.cos(Omega_tau), np.sin(Omega_tau), 0],
                        [-np.sin(Omega_tau), np.cos(Omega_tau), 0],
                        [0, 0, 1],
                    ]
                )
                data[i, :3] = data[i, :3] @ R_sagnac.T

            iteration = 0
            while (
                np.linalg.norm(delta_x) > self.estimate_epsilon
                and iteration < self.max_iterations
            ):
                for i in range(length):
                    d = np.linalg.norm(data[i, :3] - x[:3])
                    G[i, :] = [
                        (x[0] - data[i, 0]) / d,
                        (x[1] - data[i, 1]) / d,
                        (x[2] - data[i, 2]) / d,
                        1,
                    ]
                    delta_rho[i] = r[i] - d - x[3]
                delta_x = np.linalg.inv(G.T @ G) @ G.T @ delta_rho
                x += delta_x
                iteration += 1

            if iteration == self.max_iterations:
                raise RuntimeError(
                    "Maximum number of iterations reached without convergence"
                )

            truth_ecef = lla_to_ecef(truth_lla)
            estimated_lla = ecef_to_lla(Vector3(*x[:3]))
            self.estimate_lla = estimated_lla

            norm_error = np.linalg.norm(x[:3] - truth_ecef.numpy())
            horizontal_error = np.array(
                [
                    2 * np.pi * 6356909 / 360 * (estimated_lla[0] - truth_lla[0]),
                    2
                    * np.pi
                    * 6377830
                    / 360
                    * np.cos(estimated_lla[0] * np.pi / 180)
                    * (estimated_lla[1] - truth_lla[1]),
                ]
            )
            vertical_error = estimated_lla[2] - truth_lla[2]
            H = np.diag(np.linalg.inv(G.T @ G))
            sigma = np.std(delta_rho - G @ delta_x)
            GDOP = np.sqrt(np.sum(H))
            PDOP = np.sqrt(np.sum(H[:3]))
            HDOP = np.sqrt(np.sum(H[:2]))
            VDOP = np.sqrt(H[2])
            TDOP = np.sqrt(H[3])
            RMS = sigma * GDOP
            PRMS = sigma * PDOP
            HRMS = sigma * HDOP
            VRMS = sigma * VDOP
            TRMS = sigma * TDOP

            evalutaion_dict = {
                "norm_error": norm_error,
                "horizontal_error": horizontal_error,
                "vertical_error": vertical_error,
                "GDOP": GDOP,
                "PDOP": PDOP,
                "HDOP": HDOP,
                "VDOP": VDOP,
                "TDOP": TDOP,
                "RMS": RMS,
                "PRMS": PRMS,
                "HRMS": HRMS,
                "VRMS": VRMS,
                "TRMS": TRMS,
            }

            return (x, evalutaion_dict)

    def log(
        self,
        time: list,
        truth_lla: Vector3,
        satellite_info: list[dict],
        estimation_result: tuple[np.ndarray, dict],
    ):
        formatted_time = datetime(*time).strftime("%Y-%m-%d %H:%M:%S")

        formatted_time_2 = datetime(*time).strftime("%Y_%m_%d_%H_%M_%S")
        file_name = f"estimation_result_{formatted_time_2}.txt"
        x, evaluation = estimation_result
        log_path = Path(self.log_dir) / file_name
        log_path.parent.mkdir(parents=True, exist_ok=True)

        truth_ecef = lla_to_ecef(truth_lla)

        with log_path.open("w") as f:
            f.write(f"Time: {formatted_time} UTC\n")
            f.write(f"Truth Position(ECEF):\n")
            f.write(f"  X: {truth_ecef.x} meters\n")
            f.write(f"  Y: {truth_ecef.y} meters\n")
            f.write(f"  Z: {truth_ecef.z} meters\n")

            f.write(f"Truth Position(LLA):\n")
            f.write(f"  Latitude: {truth_lla[0]} degrees\n")
            f.write(f"  Longitude: {truth_lla[1]} degrees\n")
            f.write(f"  Altitude: {truth_lla[2]} meters\n")
            f.write(f"Estimated Position(ECEF):\n")
            f.write(f"  X: {x[0]} meters\n")
            f.write(f"  Y: {x[1]} meters\n")
            f.write(f"  Z: {x[2]} meters\n")

            estimated_lla = ecef_to_lla(Vector3(*x[:3]))
            f.write(f"Estimated Position(LLA):\n")
            f.write(f"  Latitude: {estimated_lla[0]} degrees\n")
            f.write(f"  Longitude: {estimated_lla[1]} degrees\n")
            f.write(f"  Altitude: {estimated_lla[2]} meters\n")
            f.write(f"Estimated Local Time Bias: {x[3]} seconds\n")
            f.write("\n")
            f.write(f"Norm Error: {evaluation['norm_error']} meters\n")
            f.write(f"Horizontal Error: {evaluation['horizontal_error']} meters\n")
            f.write(f"Vertical Error: {evaluation['vertical_error']} meters\n")
            f.write(f"GDOP: {evaluation['GDOP']}\n")
            f.write(f"PDOP: {evaluation['PDOP']}\n")
            f.write(f"HDOP: {evaluation['HDOP']}\n")
            f.write(f"VDOP: {evaluation['VDOP']}\n")
            f.write(f"TDOP: {evaluation['TDOP']}\n")
            f.write(f"RMS: {evaluation['RMS']} meters\n")
            f.write(f"PRMS: {evaluation['PRMS']} meters\n")
            f.write(f"HRMS: {evaluation['HRMS']} meters\n")
            f.write(f"VRMS: {evaluation['VRMS']} meters\n")
            f.write(f"TRMS: {evaluation['TRMS']} meters\n")
            f.write("\n")
            f.write("Satellite Information\n")
            f.write(f"Number of Satellites: {len(satellite_info)}\n")
            f.write(f"Satellite System: {satellite_info[0]['system']}\n")
            f.write("\n")
            for data in satellite_info:
                f.write(f"PRN: {data['prn']}\n")
                f.write(f"Azimuth: {data['azimuth']} radians\n")
                f.write(f"Elevation: {data['elevation']} radians\n")
                f.write(f"ENU:\n")
                f.write(f"  E: {data['enu'].x} meters\n")
                f.write(f"  N: {data['enu'].y} meters\n")
                f.write(f"  U: {data['enu'].z} meters\n")
                f.write(f"ECEF:\n")
                f.write(f"  X: {data['ecef'].x} meters\n")
                f.write(f"  Y: {data['ecef'].y} meters\n")
                f.write(f"  Z: {data['ecef'].z} meters\n")
                f.write(f"Delta T: {data['delta_t']} seconds\n")
                f.write("\n")

    def draw_skymap(
        self,
        time: list,
        satellite_info: list[dict],
        slience: bool = True,
        save: bool = True,
    ):
        formatted_time = datetime(*time).strftime("%Y-%m-%d %H:%M:%S")

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection="polar")

        for data in satellite_info:
            ax.scatter(
                data["azimuth"],
                90 - np.degrees(data["elevation"]),
                label=f"PRN: {data['prn']}",
            )
            ax.annotate(
                f"{data['prn']}",
                (data["azimuth"], 90 - np.degrees(data["elevation"])),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(90)
        ax.set_rticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
        ax.grid(True)

        plt.title(f"Satellite Skyplot {formatted_time} UTC", pad=40)
        plt.legend()

        if not slience:
            plt.show()

        if save:
            formatted_time_2 = datetime(*time).strftime("%Y_%m_%d_%H_%M_%S")
            figure_path = Path(self.figure_dir) / f"skyplot_{formatted_time_2}.png"
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(figure_path)

        plt.close(fig)


if __name__ == "__main__":
    truth = Vector3(-289833.9300, -2756501.0600, 5725162.2200)
    nav_file = "brdc2750.22n"
    obs_file = "bake2750.22o"

    estimation = PositionEstimation(truth, obs_file)
    observations = estimation.load_observation_data()
    time = observations[0][0]
    observation = observations[0][1]
    satellite_info = estimation.extract_satellite_info(time, observation, nav_file)
    estimation_result = estimation.estimate_position(satellite_info)
    estimation.draw_skymap(time, satellite_info)
    estimation.log(time, satellite_info, estimation_result)
