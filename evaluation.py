from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import Vector3
from map import Map, PathStyle


def evalution(df: pd.DataFrame, result_dir: str) -> None:
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

    log_path = Path(result_dir) / "summary_results.txt"
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


def plot_error(df: pd.DataFrame, result_dir: str, plot_flag: bool = True) -> None:
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
    plt.savefig(result_dir + "error.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_error_histograms(
    df: pd.DataFrame, result_dir: str, plot_flag: bool = True
) -> None:
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
    plt.savefig(result_dir + "error_histograms.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_gdop(df: pd.DataFrame, result_dir: str, plot_flag: bool = True) -> None:
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x="time", y="GDOP", label="GDOP")

    plt.title("GDOP Over Time")
    plt.xlabel("Time")
    plt.ylabel("GDOP")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.savefig(result_dir + "gdop.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_gdop_histogram(
    df: pd.DataFrame, result_dir: str, plot_flag: bool = True
) -> None:
    plt.figure(figsize=(10, 6))

    sns.histplot(df["GDOP"], kde=True)
    plt.title("GDOP Distribution")
    plt.xlabel("GDOP")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(result_dir + "gdop_histogram.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_vdop(df: pd.DataFrame, result_dir: str, plot_flag: bool = True) -> None:
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x="time", y="VDOP", label="VDOP")

    plt.title("Vertical Dilution of Precision (VDOP) Over Time")
    plt.xlabel("Time")
    plt.ylabel("VDOP Value")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.savefig(result_dir + "vdop.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_vdop_histogram(
    df: pd.DataFrame, result_dir: str, plot_flag: bool = True
) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(df["VDOP"], kde=True, bins=20, color="blue", label="VDOP")
    plt.title("VDOP Histogram")
    plt.xlabel("VDOP Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir + "vdop_histogram.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_hdop(df: pd.DataFrame, result_dir: str, plot_flag: bool = True) -> None:
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x="time", y="HDOP", label="HDOP")

    plt.title("Horizontal Dilution of Precision (HDOP) Over Time")
    plt.xlabel("Time")
    plt.ylabel("HDOP Value")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.savefig(result_dir + "hdop.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_hdop_histogram(
    df: pd.DataFrame, result_dir: str, plot_flag: bool = True
) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(df["HDOP"], kde=True, bins=20, color="green", label="HDOP")
    plt.title("HDOP Histogram")
    plt.xlabel("HDOP Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir + "hdop_histogram.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_pdop(df: pd.DataFrame, result_dir: str, plot_flag: bool = True) -> None:
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x="time", y="PDOP", label="PDOP")

    plt.title("Position Dilution of Precision (PDOP) Over Time")
    plt.xlabel("Time")
    plt.ylabel("PDOP Value")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.savefig(result_dir + "pdop.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_pdop_histogram(
    df: pd.DataFrame, result_dir: str, plot_flag: bool = True
) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(df["PDOP"], kde=True, bins=20, color="red", label="PDOP")
    plt.title("PDOP Histogram")
    plt.xlabel("PDOP Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir + "pdop_histogram.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_tdop(df: pd.DataFrame, result_dir: str, plot_flag: bool = True) -> None:
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x="time", y="TDOP", label="TDOP")

    plt.title("Time Dilution of Precision (TDOP) Over Time")
    plt.xlabel("Time")
    plt.ylabel("TDOP Value")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.savefig(result_dir + "tdop.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_tdop_histogram(
    df: pd.DataFrame, result_dir: str, plot_flag: bool = True
) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(df["TDOP"], kde=True, bins=20, color="purple", label="TDOP")
    plt.title("TDOP Histogram")
    plt.xlabel("TDOP Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir + "tdop_histogram.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def plot_coordinates(df: pd.DataFrame, result_dir: str, plot_flag: bool = True) -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(df["lon"], df["lat"], label="Estimated Coordinates", color="blue")
    plt.plot(df["truth_lon"], df["truth_lat"], label="True Coordinates", color="red")

    plt.title("True vs Estimated Coordinates")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.legend()

    plt.tight_layout()
    plt.savefig(result_dir + "coordinates.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


class MapPlotMode(Enum):
    SPLIT = "split"
    COMBINE = "combine"
    BOTH = "both"


def plot_map(
    df: pd.DataFrame, result_dir: str, plot_mode: MapPlotMode = MapPlotMode.SPLIT
) -> None:
    map = Map()

    estimated_locations = [
        Vector3(row["lon"], row["lat"], 0) for _, row in df.iterrows()
    ]
    true_locations = [
        Vector3(row["truth_lon"], row["truth_lat"], 0) for _, row in df.iterrows()
    ]

    path_styles = [
        PathStyle(
            weight=2,
            color="0x0000FF",
            transparency=1,
            fillcolor="",
            fillTransparency=0.5,
        ),
        PathStyle(
            weight=2,
            color="0xFF0000",
            transparency=1,
            fillcolor="",
            fillTransparency=0.5,
        ),
    ]

    estimated_uri = map.construct_paths_uri([(estimated_locations, path_styles[0])])
    true_uri = map.construct_paths_uri([(true_locations, path_styles[1])])
    combined_uri = map.construct_paths_uri(
        [
            (estimated_locations, path_styles[0]),
            (true_locations, path_styles[1]),
        ]
    )

    if plot_mode in {MapPlotMode.COMBINE, MapPlotMode.BOTH}:
        combined_map_image = map.get_map_image(combined_uri)
        with open(result_dir + "combined_map.png", "wb") as f:
            f.write(combined_map_image)

    if plot_mode in {MapPlotMode.SPLIT, MapPlotMode.BOTH}:
        estimated_map_image = map.get_map_image(estimated_uri)
        true_map_image = map.get_map_image(true_uri)
        with open(result_dir + "estimated_map.png", "wb") as f:
            f.write(estimated_map_image)
        with open(result_dir + "true_map.png", "wb") as f:
            f.write(true_map_image)


def plot_ecef_coordinates(
    df: pd.DataFrame, result_dir: str, plot_flag: bool = True
) -> None:
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

    plt.savefig(result_dir + "ecef_coordinates.png", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()


def draw_satellite_tracks(
    satellite_position: pd.DataFrame,
    result_dir: str,
    plot_flag: bool = True,
    max_gap: int = 90,
) -> None:
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="polar")
    colors = plt.colormaps["tab20"]

    for idx, prn in enumerate(satellite_position["prn"].unique()):
        sat_data = satellite_position[satellite_position["prn"] == prn]

        sat_data.loc[:, "azimuth"] = np.mod(sat_data["azimuth"], 2 * np.pi)
        sat_data.loc[:, "elevation"] = np.mod(sat_data["elevation"], 2 * np.pi)
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

    plt.savefig(result_dir + "satellite_tracks.png", dpi=300, bbox_inches="tight")

    if plot_flag:
        plt.show()
