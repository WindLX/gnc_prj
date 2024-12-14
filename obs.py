import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from rinex2 import read_rinex_nav, rinex_to_sv, rinex_to_ecef
from typing import List, Tuple
import seaborn as sns
import pandas as pd


# 光速 (单位：米/秒)
c = 299792458


# 伪距方程计算函数
def residuals(params, satellite_positions, pseudoranges):
    x, y, z, t = params  # 接收机位置(x, y, z) 和时钟偏差 t
    residuals = []
    for sat_pos, pseudorange in zip(satellite_positions, pseudoranges):
        dx = x - sat_pos[0]
        dy = y - sat_pos[1]
        dz = z - sat_pos[2]
        predicted_pseudorange = np.sqrt(dx**2 + dy**2 + dz**2) + c * t
        residuals.append(predicted_pseudorange - pseudorange)
    return np.array(residuals)


def compute_receiver_position(
    satellites: List[Tuple[float, float, float]],
    pseudoranges: List[float],
    initial_guess: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    satellite_positions = satellites
    result = least_squares(
        residuals, initial_guess, args=(satellite_positions, pseudoranges)
    )
    return result.x


class RinexPosition:
    def __init__(self, x, y, z, delta_h=0.0, delta_e=0.0, delta_n=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.delta_h = delta_h
        self.delta_e = delta_e
        self.delta_n = delta_n


class RinexHeader:
    def __init__(
        self,
        version,
        data_type,
        program,
        run_by,
        date,
        marker_name,
        marker_number,
        observer_agency,
        receiver_info,
        antenna_info,
        position: RinexPosition,
    ):
        self.version = version
        self.data_type = data_type
        self.program = program
        self.run_by = run_by
        self.date = date
        self.marker_name = marker_name
        self.marker_number = marker_number
        self.observer_agency = observer_agency
        self.receiver_info = receiver_info
        self.antenna_info = antenna_info
        self.position = position


class SatelliteObservation:
    def __init__(
        self,
        system: str,
        satellite_id: int,
        pseudorange: float,
        observations: list[float],
    ):
        self.system = system
        self.satellite_id = satellite_id
        self.pseudorange = pseudorange  # 伪距
        self.observations = observations  # 其他观测值


class RinexObservationData:
    def __init__(self, timestamp: datetime, observations: list[SatelliteObservation]):
        self.timestamp = timestamp
        self.observations = observations  # List of SatelliteObservation objects


def parse_rinex_header(header_lines) -> RinexHeader:
    # Extract general information from the header
    version = header_lines[0].split()[0]
    data_type = header_lines[0].split()[1]
    program = header_lines[1].split()[0]
    run_by = header_lines[1].split()[1]

    # Extract the correct date-time string using regex
    date_str = re.search(
        r"\d{8} \d{6}", header_lines[1]
    ).group()  # Extracts the date and time portion like '20241207 022629'
    date = datetime.strptime(date_str, "%Y%m%d %H%M%S")

    marker_name = header_lines[2].strip()
    marker_number = header_lines[3].strip()
    observer_agency = header_lines[4].strip()
    receiver_info = header_lines[5].strip()
    antenna_info = header_lines[6].strip()

    # Parse the approximate position XYZ
    position_values = list(map(float, header_lines[7].split()[:3]))
    position = RinexPosition(position_values[0], position_values[1], position_values[2])

    return RinexHeader(
        version,
        data_type,
        program,
        run_by,
        date,
        marker_name,
        marker_number,
        observer_agency,
        receiver_info,
        antenna_info,
        position,
    )


def parse_observation_data(data_lines) -> list[RinexObservationData]:
    observation_data = []

    idx = 0
    while idx < len(data_lines):
        line = data_lines[idx]
        if line.startswith(">"):
            # Parse timestamp (only take the first 6 components)
            timestamp_parts = line[1:].split()

            # Fix the timestamp: limit the fractional part of seconds to 6 digits
            time_str = " ".join(timestamp_parts[:6])
            time_str = time_str[
                : len(time_str) - len(timestamp_parts[5]) + 7
            ]  # Keep 6 digits of seconds

            # Parse the corrected timestamp string
            timestamp = datetime.strptime(time_str, "%Y %m %d %H %M %S.%f")

            # Parse observations (next lines)
            observations = []
            while True:
                try:
                    next_line = data_lines[idx + 1]
                except IndexError:
                    next_line = None
                if not next_line:
                    break
                if next_line.startswith(">"):
                    break
                idx += 1
                satellite_info = next_line.split()
                system = satellite_info[0][0]
                satellite_id = int(satellite_info[0][1:])
                obs_values = list(map(float, satellite_info[1:]))
                pseudorange = obs_values[
                    0
                ]  # Assuming the first value is the pseudorange
                observations.append(
                    SatelliteObservation(system, satellite_id, pseudorange, obs_values)
                )
            observation_data.append(RinexObservationData(timestamp, observations))
        idx += 1
    return observation_data


def parse_rinex(file_path) -> tuple[RinexHeader, list[RinexObservationData]]:
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract header and observation data
    header_lines = [
        line
        for line in lines
        if line.startswith("4.01")
        or line.startswith("GnssLogger")
        or "END OF HEADER" not in line
    ]
    header_end_index = next(
        i for i, line in enumerate(lines) if "END OF HEADER" in line
    )
    observation_lines = lines[header_end_index + 1 :]

    # Parse header
    header = parse_rinex_header(header_lines)

    # Parse observation data
    observation_data = parse_observation_data(observation_lines)

    return header, observation_data


# 解析并处理RINEX文件
def process_rinex_data(header, observation_data, rinex_data_list):
    receiver_positions = []

    for observation in observation_data:
        time = observation.timestamp
        observations = observation.observations
        position_data = {}

        print(time)

        for obs in observations:
            # 选择一个卫星
            rinex_data = next(
                (
                    data
                    for data in rinex_data_list
                    if data.svprn == obs.satellite_id and obs.system == "G"
                ),
                None,
            )

            if rinex_data:
                print(f"Satellite: {obs.satellite_id}")
                # print(f"Observations: {obs.observations}")

                # 计算卫星位置
                rinex_ecef = rinex_to_ecef(rinex_data, time)
                print(f"Satellite ECEF: {rinex_ecef}")

                position_data = {
                    "prn": obs.satellite_id,
                    "position": (rinex_ecef.x, rinex_ecef.y, rinex_ecef.z),
                    # "position": (rinex_eci[0], rinex_eci[1], rinex_eci[2]),
                }

        receiver_positions.append((time, position_data))
        print()

    return receiver_positions


if __name__ == "__main__":
    # 解析RINEX观测数据
    header, observation_data = parse_rinex("./gnss_log_2024_12_13_11_31_47.24o")

    # 打印头部信息
    print(f"RINEX Version: {header.version}, Data Type: {header.data_type}")
    print(f"Program: {header.program}, Run By: {header.run_by}, Date: {header.date}")
    print(f"Marker Name: {header.marker_name}, Marker Number: {header.marker_number}")
    print(
        f"Position: X={header.position.x}, Y={header.position.y}, Z={header.position.z}"
    )

    # 读取RINEX导航文件
    gps_rinex_file_path = "./brdc3480.24n"
    gps_rinex_head, gps_rinex_data = read_rinex_nav(gps_rinex_file_path)
    gps_rinex_data_list = gps_rinex_data

    print(len(observation_data))

    receiver_positions = []

    for obs in observation_data:
        time, observations = obs.timestamp, obs.observations
        gps_satellites = []
        for sat_obs in observations:
            gps_satellite = next(
                (
                    data
                    for data in gps_rinex_data_list
                    if data.svprn == sat_obs.satellite_id and sat_obs.system == "G"
                ),
                None,
            )
            if gps_satellite:
                gps_satellites.append(
                    (sat_obs.satellite_id, gps_satellite, sat_obs.pseudorange)
                )
        print(time)

        ecef_positions = []
        for id, gps, pseudorange in gps_satellites:
            # 计算卫星位置
            rinex_ecef = rinex_to_ecef(gps, time)
            ecef_positions.append((id, rinex_ecef, pseudorange))
            print(f"ID: {id}, {rinex_ecef}, {pseudorange}")

        # 计算接收机位置
        if len(ecef_positions) >= 4:
            satellites = [(pos[1].x, pos[1].y, pos[1].z) for pos in ecef_positions]
            pseudoranges = [pos[2] for pos in ecef_positions]
            initial_guess = (header.position.x, header.position.y, header.position.z, 0)
            receiver_position = compute_receiver_position(
                satellites, pseudoranges, initial_guess
            )
            receiver_positions.append((time, receiver_position))
            print(f"Receiver Position: {receiver_position}")
        print()

    # 绘制接收机位置
    # Prepare data for plotting
    times = [pos[0] for pos in receiver_positions]
    times = times[30:60]
    x_coords = [pos[1][0] for pos in receiver_positions]
    y_coords = [pos[1][1] for pos in receiver_positions]
    z_coords = [pos[1][2] for pos in receiver_positions]
    x_coords = x_coords[30:60]
    y_coords = y_coords[30:60]
    z_coords = z_coords[30:60]

    # Create a DataFrame for seaborn
    data = pd.DataFrame({"Time": times, "X": x_coords, "Y": y_coords, "Z": z_coords})

    # Plot the receiver position over time in three subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    sns.lineplot(ax=axes[0], x="Time", y="X", data=data, marker="o")
    axes[0].set_title("Receiver X Coordinate Over Time")
    axes[0].set_ylabel("X Coordinate")

    sns.lineplot(ax=axes[1], x="Time", y="Y", data=data, marker="o")
    axes[1].set_title("Receiver Y Coordinate Over Time")
    axes[1].set_ylabel("Y Coordinate")

    sns.lineplot(ax=axes[2], x="Time", y="Z", data=data, marker="o")
    axes[2].set_title("Receiver Z Coordinate Over Time")
    axes[2].set_ylabel("Z Coordinate")
    axes[2].set_xlabel("Time")

    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     header, observation_data = parse_rinex("./gnss_log_2024_12_07_10_26_10.24o")

#     # Print parsed header information
#     print(f"RINEX Version: {header.version}, Data Type: {header.data_type}")
#     print(f"Program: {header.program}, Run By: {header.run_by}, Date: {header.date}")
#     print(f"Marker Name: {header.marker_name}, Marker Number: {header.marker_number}")
#     print(
#         f"Position: X={header.position.x}, Y={header.position.y}, Z={header.position.z}"
#     )

#     rinex_file_path = "./brdc3420.24n"
#     rinex_head, rinex_data = read_rinex_nav(rinex_file_path)
#     rinex_data_list = list(filter(lambda x: x.svprn == 23, rinex_data))

#     rinex_coords = []

#     for observation in observation_data:
#         print(f"Timestamp: {observation.timestamp}")
#         for sat_obs in observation.observations:
#             print(
#                 f"System: {sat_obs.system}, Satellite ID: {sat_obs.satellite_id}, Observations: {sat_obs.observations}"
#             )

#             # Find corresponding RINEX data for the satellite
#             rinex_data = next(
#                 (
#                     data
#                     for data in rinex_data_list
#                     if data.svprn == sat_obs.satellite_id
#                 ),
#                 None,
#             )
#             if rinex_data:
#                 print("RINEX Data:")
#                 print(rinex_data)
#                 date = rinex_data.date
#                 print()

#                 rinex_ecef = rinex_to_ecef(rinex_data, observation.timestamp)
#                 print("ECEF Data from RINEX:")
#                 print(rinex_ecef)
#                 print()

#                 rinex_coords.append((rinex_ecef.x, rinex_ecef.y, rinex_ecef.z))

#     if rinex_coords:
#         rinex_x, rinex_y, rinex_z = zip(*rinex_coords)

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")

#     ax.plot(rinex_x, rinex_y, rinex_z, label="RINEX ECEF", color="blue")

#     ax.set_xlabel("X Coordinate")
#     ax.set_ylabel("Y Coordinate")
#     ax.set_zlabel("Z Coordinate")
#     ax.set_title("ECEF Coordinates from YUMA and RINEX")
#     ax.legend()

#     plt.show()
