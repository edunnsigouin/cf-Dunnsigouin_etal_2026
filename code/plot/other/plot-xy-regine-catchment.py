"""
Plots the border of an nve regine catchement. You can compare with those found here:
https://temakart.nve.no/tema/nedborfelt
"""

import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from Dunnsigouin_etal_2026 import config


# input -------------------------------------------------------
path_in      = config.dirs["nve_catchment"]
path_out     = config.dirs["fig"]
filename_in  = f"{path_in}nve_regine_enhet_012_drammensvassdraget_entire_catchment.geojson"
filename_out = f"{path_out}nve_regine_enhet_012_drammensvassdraget_entire_catchment.pdf"
map_extent   = [4.5, 14.0, 57.5, 64.0]   # lon_min, lon_max, lat_min, lat_max
line_width   = 2.0 # plot styling
figure_size  = (6, 6)
write2file   = True
# -------------------------------------------------------------


def read_geojson(filepath: str) -> dict:
    """Read a GeoJSON file into a Python dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_cartopy_map(fig_size, extent):
    """Create a Cartopy axis and add simple geographic context layers."""
    proj = ccrs.LambertConformal(
        central_longitude=15,
        central_latitude=65,
        standard_parallels=(63, 70),
    )

    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(projection=proj)

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return fig, ax


def plot_ring(ax, ring_coords, linewidth):
    """Plot a single ring (list of [lon,lat]) as a line."""
    lons = [pt[0] for pt in ring_coords]
    lats = [pt[1] for pt in ring_coords]
    ax.plot(
        lons,
        lats,
        linewidth=linewidth,
        color="tab:red",
        transform=ccrs.PlateCarree(),
    )


def plot_polygon_border(ax, geometry: dict, linewidth: float):
    """
    Plot ONLY the outer border(s) from a Polygon/MultiPolygon GeoJSON geometry.

    Supports:
      - Polygon: coords = [exterior, hole1, hole2, ...]
      - MultiPolygon: coords = [poly1, poly2, ...] each poly = [exterior, holes...]
    """
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")

    if gtype == "Polygon":
        exterior = coords[0]
        plot_ring(ax, exterior, linewidth)

    elif gtype == "MultiPolygon":
        for poly in coords:
            exterior = poly[0]
            plot_ring(ax, exterior, linewidth)

    else:
        raise ValueError(
            f"Expected Polygon/MultiPolygon from dissolved catchment file, got: {gtype}"
        )


def add_gridlines(ax):
    """Add labeled gridlines to the map."""
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False



if __name__ == "__main__":

    # Read GeoJSON (dissolved polygon catchment)
    gj = read_geojson(filename_in)
    features = gj.get("features", [])
    if not features:
        raise ValueError("No features found in input GeoJSON.")

    feature = features[0]
    geometry = feature.get("geometry", {})
    properties = feature.get("properties", {})

    # Set up map
    fig, ax = setup_cartopy_map(figure_size, map_extent)

    # Plot border(s) of the dissolved polygon(s)
    plot_polygon_border(ax, geometry, line_width)

    # Add gridlines and title
    add_gridlines(ax)
    title = properties.get("source", "NVE REGINE catchment")
    ax.set_title(title, fontsize=12)

    if write2file:
        fig.savefig(filename_out, bbox_inches="tight")
        print("Saved:", filename_out)

    plt.show()
