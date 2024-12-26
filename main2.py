from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from model import Vector3
from utils import ecef_to_lla
from estimation import PositionEstimation


def evalution(df):
    mean_position = df[["ecef_x", "ecef_y", "ecef_z"]].mean()
    variance_position = df[["ecef_x", "ecef_y", "ecef_z"]].var()

    mean_errors = df[
        [
            "norm_error",
            "horizontal_error_lat",
            "horizontal_error_lon",
            "vertical_error",
        ]
    ].mean()
    variance_errors = df[
        [
            "norm_error",
            "horizontal_error_lat",
            "horizontal_error_lon",
            "vertical_error",
        ]
    ].var()

    print("Mean Estimated Position (ECEF):")
    print(mean_position)
    print("\nVariance of Estimated Position (ECEF):")
    print(variance_position)

    print("\nMean Errors:")
    print(mean_errors)
    print("\nVariance of Errors:")
    print(variance_errors)

    gdop_mean = df["GDOP"].mean()
    print(f"\nMean GDOP: {gdop_mean}")
    gdop_variance = df["GDOP"].var()
    print(f"Variance of GDOP: {gdop_variance}")

    log_path = Path("log") / "summary_results.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as f:
        f.write("Mean Estimated Position (ECEF):\n")
        f.write(f"  X: {mean_position['ecef_x']} meters\n")
        f.write(f"  Y: {mean_position['ecef_y']} meters\n")
        f.write(f"  Z: {mean_position['ecef_z']} meters\n")
        f.write("\nVariance of Estimated Position (ECEF):\n")
        f.write(f"  X: {variance_position['ecef_x']} meters^2\n")
        f.write(f"  Y: {variance_position['ecef_y']} meters^2\n")
        f.write(f"  Z: {variance_position['ecef_z']} meters^2\n")
        f.write("\nMean Errors:\n")
        f.write(f"  Norm Error: {mean_errors['norm_error']} meters\n")
        f.write(
            f"  Horizontal Error Latitude: {mean_errors['horizontal_error_lat']} meters\n"
        )
        f.write(
            f"  Horizontal Error Longitude: {mean_errors['horizontal_error_lon']} meters\n"
        )
        f.write(f"  Vertical Error: {mean_errors['vertical_error']} meters\n")
        f.write("\nVariance of Errors:\n")
        f.write(f"  Norm Error: {variance_errors['norm_error']} meters^2\n")
        f.write(
            f"  Horizontal Error Latitude: {variance_errors['horizontal_error_lat']} meters^2\n"
        )
        f.write(
            f"  Horizontal Error Longitude: {variance_errors['horizontal_error_lon']} meters^2\n"
        )
        f.write(f"  Vertical Error: {variance_errors['vertical_error']} meters^2\n")
        f.write("\nMean GDOP:\n")
        f.write(f"  {gdop_mean}\n")
        f.write("\nVariance of GDOP:\n")
        f.write(f"  {gdop_variance}\n")


def plot_error(df):
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="time",
        y="norm_error",
        label="Norm Error",
    )

    sns.lineplot(
        data=df,
        x="time",
        y="horizontal_error_lat",
        label="Horizontal Error Latitude",
    )

    sns.lineplot(
        data=df,
        x="time",
        y="horizontal_error_lon",
        label="Horizontal Error Longitude",
    )

    sns.lineplot(data=df, x="time", y="vertical_error", label="Vertical Error")

    plt.title("Position Estimation Errors Over Time")
    plt.xlabel("Time")
    plt.ylabel("Error (meters)")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_error_histograms(df):
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    sns.histplot(df["norm_error"], kde=True)
    plt.title("Norm Error Distribution")
    plt.xlabel("Norm Error (meters)")

    plt.subplot(2, 2, 2)
    sns.histplot(df["horizontal_error_lat"], kde=True)
    plt.title("Horizontal Error Latitude Distribution")
    plt.xlabel("Horizontal Error Latitude (meters)")

    plt.subplot(2, 2, 3)
    sns.histplot(df["horizontal_error_lon"], kde=True)
    plt.title("Horizontal Error Longitude Distribution")
    plt.xlabel("Horizontal Error Longitude (meters)")

    plt.subplot(2, 2, 4)
    sns.histplot(df["vertical_error"], kde=True)
    plt.title("Vertical Error Distribution")
    plt.xlabel("Vertical Error (meters)")

    plt.tight_layout()
    plt.show()


def plot_gdop(df):
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x="time", y="GDOP", label="GDOP")

    plt.title("GDOP Over Time")
    plt.xlabel("Time")
    plt.ylabel("GDOP")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_gdop_histogram(df):
    plt.figure(figsize=(10, 6))

    sns.histplot(df["GDOP"], kde=True)
    plt.title("GDOP Distribution")
    plt.xlabel("GDOP")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_coordinates(df):
    plt.figure(figsize=(10, 6))

    plt.plot(df["lon"], df["lat"], label="Estimated Coordinates", color="blue")
    plt.plot(df["truth_lon"], df["truth_lat"], label="True Coordinates", color="red")

    plt.title("True vs Estimated Coordinates")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_ecef_coordinates(df):
    plt.figure(figsize=(14, 8))

    plt.subplot(3, 1, 1)
    sns.lineplot(data=df, x="time", y="ecef_x", label="ECEF X")
    plt.title("ECEF X Coordinate Over Time")
    plt.xlabel("Time")
    plt.ylabel("ECEF X (meters)")

    plt.subplot(3, 1, 2)
    sns.lineplot(data=df, x="time", y="ecef_y", label="ECEF Y")
    plt.title("ECEF Y Coordinate Over Time")
    plt.xlabel("Time")
    plt.ylabel("ECEF Y (meters)")

    plt.subplot(3, 1, 3)
    sns.lineplot(data=df, x="time", y="ecef_z", label="ECEF Z")
    plt.title("ECEF Z Coordinate Over Time")
    plt.xlabel("Time")
    plt.ylabel("ECEF Z (meters)")

    plt.tight_layout()
    plt.show()


def draw_satellite_tracks(satellite_position, max_gap=90):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="polar")
    colors = plt.cm.get_cmap("tab20", len(satellite_position["prn"].unique()))

    for idx, prn in enumerate(satellite_position["prn"].unique()):
        sat_data = satellite_position[satellite_position["prn"] == prn]

        sat_data["azimuth"] = np.mod(sat_data["azimuth"], 2 * np.pi)
        sat_data["elevation"] = np.mod(sat_data["elevation"], 2 * np.pi)
        azimuth_diff = np.diff(sat_data["azimuth"])
        elevation_diff = np.diff(sat_data["elevation"])
        large_changes = (np.abs(azimuth_diff) > np.radians(max_gap)) | (
            np.abs(elevation_diff) > np.radians(max_gap)
        )
        segments = np.split(sat_data, np.where(large_changes)[0] + 1)

        for segment in segments:
            ax.plot(
                segment["azimuth"],
                90 - np.degrees(segment["elevation"]),
                color=colors(idx),
                label=f"PRN {prn}" if segment.index[0] == sat_data.index[0] else "",
            )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.set_rticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
    ax.grid(True)

    plt.title("Satellite Tracks on Skyplot", pad=40)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    nav_file = "./nav/brdc3490.24n"

    file_index = 1
    obs_files = [
        "./data/1_Medium_Interference_Near_SAA_1525/gnss_log_2024_12_14_15_17_38.24o",
        "./data/2_Open_field_Lawn_1541/gnss_log_2024_12_14_15_28_53.24o",
        "./data/3_Forest_1542/gnss_log_2024_12_14_15_46_31.24o",
        "./data/4_LibraryBasement_1606/gnss_log_2024_12_14_16_02_14.24o",
        "./data/5_DormArea_HighInterference_1620/gnss_log_2024_12_14_16_14_30.24o",
        "./data/6_FootBall Field_1639/gnss_log_2024_12_14_16_32_45.24o",
        "./data/7_WayBackToSAA_1659/gnss_log_2024_12_14_16_49_34.24o",
    ]
    obs_file = obs_files[file_index]

    track_files = [
        "./data/1_Medium_Interference_Near_SAA_1525/doc.kml",
        "./data/2_Open_field_Lawn_1541/doc.kml",
        "./data/3_Forest_1542/doc.kml",
        "./data/4_LibraryBasement_1606/doc.kml",
        "./data/5_DormArea_HighInterference_1620/doc.kml",
        "./data/6_FootBall Field_1639/doc.kml",
        "./data/7_WayBackToSAA_1659/doc.kml",
    ]
    track_file = track_files[file_index]

    estimation = PositionEstimation(obs_file, track_file, step=1)
    observations = estimation.load_observation_data()

    evaluation_results = {
        "time": [],
        "norm_error": [],
        "horizontal_error_lat": [],
        "horizontal_error_lon": [],
        "vertical_error": [],
        "ecef_x": [],
        "ecef_y": [],
        "ecef_z": [],
        "lat": [],
        "lon": [],
        "alt": [],
        "truth_lat": [],
        "truth_lon": [],
        "truth_alt": [],
        "GDOP": [],
    }

    satellite_position = None

    for time, observation in tqdm(observations, desc="Processing Observations"):
        satellite_info, truth_lla = estimation.extract_satellite_info(
            time, observation, nav_file
        )
        try:
            estimation_result = estimation.estimate_position(truth_lla, satellite_info)
        except ValueError as e:
            print(f"Error: {e}")
            continue
        except RuntimeError as e:
            print(f"Error: {e}")
            continue

        estimation.draw_skymap(time, satellite_info)
        estimation.log(time, truth_lla, satellite_info, estimation_result)

        evaluation_results["time"].append(datetime(*time))
        evaluation_results["norm_error"].append(estimation_result[1]["norm_error"])
        evaluation_results["horizontal_error_lat"].append(
            estimation_result[1]["horizontal_error"][0]
        )
        evaluation_results["horizontal_error_lon"].append(
            estimation_result[1]["horizontal_error"][1]
        )
        evaluation_results["vertical_error"].append(
            estimation_result[1]["vertical_error"]
        )

        evaluation_results["GDOP"].append(estimation_result[1]["GDOP"])

        ecef_position = estimation_result[0]
        evaluation_results["ecef_x"].append(ecef_position[0])
        evaluation_results["ecef_y"].append(ecef_position[1])
        evaluation_results["ecef_z"].append(ecef_position[2])

        lla_position = ecef_to_lla(Vector3.from_list(ecef_position))
        evaluation_results["lat"].append(lla_position[0])
        evaluation_results["lon"].append(lla_position[1])
        evaluation_results["alt"].append(lla_position[2])

        evaluation_results["truth_lat"].append(truth_lla[0])
        evaluation_results["truth_lon"].append(truth_lla[1])
        evaluation_results["truth_alt"].append(truth_lla[2])

        for sat_info in satellite_info:
            new_row = pd.DataFrame(
                {
                    "time": [datetime(*time)],
                    "prn": [sat_info["prn"]],
                    "azimuth": [sat_info["azimuth"]],
                    "elevation": [sat_info["elevation"]],
                }
            )
            if satellite_position is None:
                satellite_position = new_row
            else:
                satellite_position = pd.concat(
                    [satellite_position, new_row], ignore_index=True
                )

    df = pd.DataFrame(evaluation_results)
    sns.set_theme(style="whitegrid")
    evalution(df)
    # plot_error(df)
    # plot_error_histograms(df)
    plot_coordinates(df)
    plot_ecef_coordinates(df)
    plot_gdop(df)
    plot_gdop_histogram(df)
    draw_satellite_tracks(satellite_position)
