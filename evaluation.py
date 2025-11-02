"""
Evaluation module for GNSS trajectory analysis.
Computes metrics and generates all required plots (errors, coordinates, DOPs, satellites, map).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union
from pathlib import Path
from enum import Enum
import io
import logging

# Import Map for plotting
from map import Map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MapPlotMode(Enum):
    """Enum for plot modes: true trajectory, estimated, or both."""
    TRUE = "true"
    EST = "est"
    BOTH = "both"

def _ensure_path(result_dir: Union[str, Path]) -> Path:
    """Helper: Ensure result_dir is a Path object."""
    return Path(result_dir) if isinstance(result_dir, str) else result_dir

def evaluation(df: pd.DataFrame, result_dir: Union[str, Path]):
    """
    Main evaluation runner: Computes and prints key metrics.
    """
    result_dir = _ensure_path(result_dir)
    result_dir.mkdir(exist_ok=True, parents=True)
    
    if df.empty:
        logger.warning("Empty DataFrame; skipping evaluation.")
        return
    
    # Basic metrics
    norm_error = df['norm_error'].mean() if 'norm_error' in df else 0
    horizontal_error = np.sqrt(df['horizontal_error_lat']**2 + df['horizontal_error_lon']**2).mean() if 'horizontal_error_lat' in df else 0
    vertical_error = df['vertical_error'].mean() if 'vertical_error' in df else 0
    
    print(f"Average Norm Error: {norm_error:.2f}m")
    print(f"Average Horizontal Error: {horizontal_error:.2f}m")
    print(f"Average Vertical Error: {vertical_error:.2f}m")
    
    # Save summary CSV
    summary_path = result_dir / "evaluation_summary.csv"
    summary_df = pd.DataFrame({
        'metric': ['norm_error_mean', 'horizontal_error_mean', 'vertical_error_mean'],
        'value': [norm_error, horizontal_error, vertical_error]
    })
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

def plot_error(df: pd.DataFrame, result_dir: Union[str, Path], plot_flag: bool = False):
    """Plot norm, horizontal, and vertical errors over time."""
    result_dir = _ensure_path(result_dir)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['time'], df['norm_error'], label='Norm Error', alpha=0.7)
    ax.plot(df['time'], np.sqrt(df['horizontal_error_lat']**2 + df['horizontal_error_lon']**2), label='Horizontal Error', alpha=0.7)
    ax.plot(df['time'], df['vertical_error'], label='Vertical Error', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Error (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.title('Position Errors Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = result_dir / "position_errors.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if plot_flag:
        plt.show()
    plt.close()
    print(f"Error plot saved to {save_path}")

def plot_error_histograms(df: pd.DataFrame, result_dir: Union[str, Path], plot_flag: bool = False):
    """Histogram of error distributions."""
    result_dir = _ensure_path(result_dir)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    errors = ['norm_error', 'horizontal_error_lat', 'horizontal_error_lon']
    for i, err in enumerate(errors):
        if err in df:
            sns.histplot(df[err], kde=True, ax=axes[i])
            axes[i].set_title(f'{err.replace("_", " ").title()} Histogram')
    plt.tight_layout()
    save_path = result_dir / "error_histograms.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if plot_flag:
        plt.show()
    plt.close()
    print(f"Error histograms saved to {save_path}")

def plot_coordinates(df: pd.DataFrame, result_dir: Union[str, Path], plot_flag: bool = False):
    """Plot LLA coordinates (true vs est)."""
    result_dir = _ensure_path(result_dir)
    if df.empty or 'lat' not in df:
        logger.warning("Missing lat; skipping coordinates plot.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(df['time'], df['lat'], label='Est Lat', alpha=0.7)
    ax1.plot(df['time'], df['truth_lat'], label='True Lat', alpha=0.7)
    ax1.set_ylabel('Latitude')
    ax1.legend()
    ax2.plot(df['time'], df['lon'], label='Est Lon', alpha=0.7)
    ax2.plot(df['time'], df['truth_lon'], label='True Lon', alpha=0.7)
    ax2.set_ylabel('Longitude')
    ax2.legend()
    plt.suptitle('LLA Coordinates Over Time')
    plt.tight_layout()
    save_path = result_dir / "lla_coordinates.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if plot_flag:
        plt.show()
    plt.close()
    print(f"Coordinates plot saved to {save_path}")

def plot_ecef_coordinates(df: pd.DataFrame, result_dir: Union[str, Path], plot_flag: bool = False):
    """Plot ECEF coordinates over time."""
    result_dir = _ensure_path(result_dir)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ecef_cols = ['ecef_x', 'ecef_y', 'ecef_z']
    for i, col in enumerate(ecef_cols):
        if col in df:
            axes[i].plot(df['time'], df[col], alpha=0.7)
            axes[i].set_ylabel(col.upper())
            axes[i].grid(True, alpha=0.3)
    plt.suptitle('ECEF Coordinates Over Time')
    plt.tight_layout()
    save_path = result_dir / "ecef_coordinates.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if plot_flag:
        plt.show()
    plt.close()
    print(f"ECEF plot saved to {save_path}")

# DOP Plots (generic helpers)
def _plot_dop(df: pd.DataFrame, dop_col: str, title: str, result_dir: Union[str, Path], plot_flag: bool = False):
    """Generic DOP line plot."""
    result_dir = _ensure_path(result_dir)
    if dop_col not in df:
        logger.warning(f"Missing {dop_col}; skipping.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['time'], df[dop_col], alpha=0.7)
    ax.set_ylabel('DOP Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = result_dir / f"{dop_col}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if plot_flag:
        plt.show()
    plt.close()
    print(f"{title} plot saved to {save_path}")

plot_gdop = lambda df, result_dir, plot_flag=False: _plot_dop(df, 'GDOP', 'GDOP Over Time', result_dir, plot_flag)
plot_vdop = lambda df, result_dir, plot_flag=False: _plot_dop(df, 'VDOP', 'VDOP Over Time', result_dir, plot_flag)
plot_hdop = lambda df, result_dir, plot_flag=False: _plot_dop(df, 'HDOP', 'HDOP Over Time', result_dir, plot_flag)
plot_pdop = lambda df, result_dir, plot_flag=False: _plot_dop(df, 'PDOP', 'PDOP Over Time', result_dir, plot_flag)
plot_tdop = lambda df, result_dir, plot_flag=False: _plot_dop(df, 'TDOP', 'TDOP Over Time', result_dir, plot_flag)

# DOP Histograms
def _plot_dop_histogram(df: pd.DataFrame, dop_col: str, title: str, result_dir: Union[str, Path], plot_flag: bool = False):
    """Generic DOP histogram."""
    result_dir = _ensure_path(result_dir)
    if dop_col not in df:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df[dop_col], kde=True, ax=ax)
    ax.set_title(f'{title} Histogram')
    plt.tight_layout()
    save_path = result_dir / f"{dop_col}_histogram.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if plot_flag:
        plt.show()
    plt.close()
    print(f"{title} histogram saved to {save_path}")

plot_gdop_histogram = lambda df, result_dir, plot_flag=False: _plot_dop_histogram(df, 'GDOP', 'GDOP', result_dir, plot_flag)
plot_vdop_histogram = lambda df, result_dir, plot_flag=False: _plot_dop_histogram(df, 'VDOP', 'VDOP', result_dir, plot_flag)
plot_hdop_histogram = lambda df, result_dir, plot_flag=False: _plot_dop_histogram(df, 'HDOP', 'HDOP', result_dir, plot_flag)
plot_pdop_histogram = lambda df, result_dir, plot_flag=False: _plot_dop_histogram(df, 'PDOP', 'PDOP', result_dir, plot_flag)
plot_tdop_histogram = lambda df, result_dir, plot_flag=False: _plot_dop_histogram(df, 'TDOP', 'TDOP', result_dir, plot_flag)

def draw_satellite_tracks(satellite_position: pd.DataFrame, result_dir: Union[str, Path], plot_flag: bool = False):
    """Plot satellite azimuth/elevation tracks."""
    result_dir = _ensure_path(result_dir)
    if satellite_position is None or satellite_position.empty:
        logger.warning("No satellite data; skipping tracks plot.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=satellite_position, x='azimuth', y='elevation', hue='prn', ax=ax1)
    ax1.set_title('Satellite Positions (Az/El)')
    ax1.set_xlabel('Azimuth')
    ax1.set_ylabel('Elevation')
    
    # Tracks by PRN
    for prn in satellite_position['prn'].unique():
        prn_data = satellite_position[satellite_position['prn'] == prn].sort_values('time')
        ax2.plot(prn_data['azimuth'], prn_data['elevation'], label=f'PRN {prn}', alpha=0.7)
    ax2.set_title('Satellite Tracks')
    ax2.set_xlabel('Azimuth')
    ax2.set_ylabel('Elevation')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_path = result_dir / "satellite_tracks.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if plot_flag:
        plt.show()
    plt.close()
    print(f"Satellite tracks saved to {save_path}")

def plot_map(df: pd.DataFrame, result_dir: Union[str, Path], plot_mode: Union[MapPlotMode, str] = MapPlotMode.BOTH,
             map_provider: str = "osm", map_style: str = "terrain") -> None:
    """Plot GNSS trajectories on map with fallback."""
    result_dir = _ensure_path(result_dir)
    if isinstance(plot_mode, str):
        try:
            plot_mode = MapPlotMode(plot_mode.upper())
        except ValueError:
            plot_mode = MapPlotMode.BOTH
    
    if df.empty:
        logger.warning("Empty df; skipping map plot.")
        return
    
    # Bounds
    lats = pd.concat([df['truth_lat'], df['lat']])
    lons = pd.concat([df['truth_lon'], df['lon']])
    min_lat, max_lat = lats.min(), lats.max()
    min_lon, max_lon = lons.min(), lons.max()
    padding = 0.01
    min_lat, max_lat = min_lat - padding, max_lat + padding
    min_lon, max_lon = min_lon - padding, max_lon + padding
    
    map_fetcher = Map(provider=map_provider, style=map_style)
    zoom = 13
    center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
    tile_x, tile_y = map_fetcher._latlon_to_tile(center_lat, center_lon, zoom)
    true_uri = map_fetcher.get_tile_uri(zoom, tile_x, tile_y)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('GNSS Trajectory Comparison')
    ax.grid(True, alpha=0.3)
    
    # Background fallback
    true_map_image = map_fetcher.get_map_image(true_uri)
    height, width = 800, 800
    if true_map_image is not None:
        try:
            true_map_img = plt.imread(io.BytesIO(true_map_image))
            ax.imshow(true_map_img, extent=[min_lon, max_lon, min_lat, max_lat], aspect='equal')
        except Exception as e:
            logger.warning(f"Image load error: {e}. Using blank.")
    if true_map_image is None:
        blank_image = np.ones((height, width, 3))
        ax.imshow(blank_image, extent=[min_lon, max_lon, min_lat, max_lat], aspect='equal')
        ax.text(0.5, 0.98, 'Map tiles unavailable (network/offline fallback)', 
                transform=ax.transAxes, ha='center', va='top', fontsize=10, color='gray')
    
    # Trajectories
    if plot_mode in [MapPlotMode.TRUE, MapPlotMode.BOTH]:
        ax.plot(df['truth_lon'], df['truth_lat'], 'g-', label='True Trajectory', linewidth=2, alpha=0.8)
        ax.scatter(df['truth_lon'], df['truth_lat'], c='green', s=20, alpha=0.6)
    
    if plot_mode in [MapPlotMode.EST, MapPlotMode.BOTH]:
        ax.plot(df['lon'], df['lat'], 'r--', label='Est Trajectory', linewidth=2, alpha=0.8)
        ax.scatter(df['lon'], df['lat'], c='red', s=20, alpha=0.6)
    
    ax.legend()
    plt.tight_layout()
    save_path = result_dir / "gnss_trajectory_map.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # No show for batch
    print(f"Map plot saved to {save_path}")