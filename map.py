"""
Map utility for GNSS trajectory plotting.
Handles fetching map tile images from providers like OpenStreetMap or Stamen.
Robust to network issues with retries and fallbacks.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional
import logging

# Configure logging if not set elsewhere
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Map:
    """
    Map image fetcher.
    Supports static map tile retrieval for overlaying GNSS trajectories.
    """

    def __init__(self, provider: str = "osm", style: str = "terrain"):
        """
        Initialize the Map fetcher.

        Args:
            provider: Map provider ('osm' for OpenStreetMap, 'stamen' for Stamen).
            style: Style for Stamen ('terrain', 'toner', 'watercolor').
        """
        self.provider = provider
        self.style = style
        self.base_url = self._get_base_url()

    def _get_base_url(self) -> str:
        """Get base URL based on provider."""
        if self.provider == "osm":
            return "https://tile.openstreetmap.org"
        elif self.provider == "stamen":
            return f"https://stamen-tiles-{self.style}.a.ssl.fastly.net"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def get_tile_uri(self, zoom: int, x: int, y: int, scale: str = "", fmt: str = ".png") -> str:
        """
        Generate URI for a single map tile.

        Args:
            zoom: Zoom level (0-19).
            x, y: Tile coordinates.
            scale: Scale factor (e.g., '@2x').
            fmt: Image format.

        Returns:
            Full URI string.
        """
        if self.provider == "osm":
            return f"{self.base_url}/{zoom}/{x}/{y}{fmt}"
        elif self.provider == "stamen":
            return f"{self.base_url}/{zoom}/{x}/{y}{scale}{fmt}"
        raise ValueError(f"Unsupported provider: {self.provider}")

    def get_map_image(self, uri: str) -> Optional[bytes]:
        """
        Fetch map image from the given URI with retries and error handling.

        Args:
            uri: Full URL to the map tile/image.

        Returns:
            Image bytes if successful, None on failure (for graceful fallback in plotting).
        """
        logger.info(f"Fetching map from: {uri}")
        
        session = requests.Session()
        retry_strategy = Retry(
            total=3,  # Max 3 retries
            backoff_factor=1,  # Exponential backoff: 1s, 2s, 4s
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on common server errors
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        try:
            response = session.get(
                uri,
                timeout=30,  # 30-second timeout to prevent hangs
                headers={
                    "User-Agent": "GNSS-GNC-Project/1.0[](https://github.com/WindLX/gnc_prj)"
                }  # Polite User-Agent to avoid rate-limiting
            )
            response.raise_for_status()  # Raise exception for bad status codes (4xx/5xx)
            logger.info("Map image fetched successfully.")
            return response.content  # Raw bytes for use with plt.imread or similar
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection reset/error fetching map: {e}. Skipping background.")
            return None
        except requests.exceptions.Timeout:
            logger.error("Timeout fetching map. Skipping background.")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for map: {e}. Skipping background.")
            return None
        finally:
            session.close()

    def get_static_map_image(self, lat: float, lon: float, zoom: int = 13, width: int = 800, height: int = 600) -> Optional[bytes]:
        """
        Fetch a static map image centered on lat/lon (alternative to tiles for simple bounds).

        Note: This uses a hypothetical static API; adapt for your provider (e.g., Google Static Maps requires key).
        For OSM/Stamen, use tile stitching or this as a placeholder.

        Args:
            lat, lon: Center coordinates.
            zoom: Zoom level.
            width, height: Image dimensions.

        Returns:
            Image bytes or None.
        """
        # Placeholder: Generate a tile-based URI or use a static endpoint
        # Example for OpenStreetMap static (requires external service or stitching)
        # For now, compute tile coords and fetch one central tile
        tile_x, tile_y = self._latlon_to_tile(lat, lon, zoom)
        uri = self.get_tile_uri(zoom, tile_x, tile_y)
        return self.get_map_image(uri)

    def _latlon_to_tile(self, lat: float, lon: float, zoom: int) -> tuple[int, int]:
        """
        Convert lat/lon to tile x,y (Mercator projection).

        Args:
            lat, lon: Geographic coordinates.
            zoom: Zoom level.

        Returns:
            (x, y) tile indices.
        """
        import math
        n = 2 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y


# Example usage (for testing; remove in production)
if __name__ == "__main__":
    map_fetcher = Map(provider="osm")
    # Test fetch
    uri = map_fetcher.get_tile_uri(zoom=10, x=572, y=361)  # Example tile for Delhi area
    image = map_fetcher.get_map_image(uri)
    if image:
        print(f"Fetched {len(image)} bytes")
    else:
        print("Fetch failed (expected if network issue)")