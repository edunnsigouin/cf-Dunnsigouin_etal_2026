"""
Plot the border of a NVE REGINE catchment GeoJSON using Cartopy.
The catchment should be the same as one of the largest found here:
https://temakart.nve.no/tema/nedborfelt
"""

import json
import matplotlib.pyplot     as plt
import cartopy.crs           as ccrs
import cartopy.feature       as cfeature
from   Dunnsigouin_etal_2026 import config


# user-defined input parameters -----------------------------------------------
path_in      = config.dirs["nve_catchment"]
path_out     = config.dirs['fig']
filename_in  = f"{path_in}nve_regine_enhet_002_glommavassdraget_entire_catchment.geojson"
filename_out = f"{path_out}xy_nve_regine_enhet_002_glommavassdraget_entire_catchment.pdf"
map_extent   = [4.5, 14.0, 57.5, 64.0]   # lon_min, lon_max, lat_min, lat_max
line_width   = 2.0 # plot styling
figure_size  = (8, 8)
write2file   = True
# -----------------------------------------------------------------------------

def read_geojson(filepath: str) -> dict:
    """Read a GeoJSON file into a Python dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_cartopy_map(fig_size, extent):
    """
    Create a Cartopy axis with a Norway-friendly Lambert Conformal projection
    and add basic background features.
    """
    proj = ccrs.LambertConformal(
        central_longitude=15,
        central_latitude=65,
        standard_parallels=(63, 70),
    )

    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(projection=proj)

    # Background layers
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3)

    # Set map extent in lon/lat
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    return fig, ax


def plot_line_like(ax, coords, linewidth):
    """
    Plot any 'line-like' GeoJSON coordinate array:
    - LineString coords: [[lon, lat], ...]
    - LinearRing coords: [[lon, lat], ...] (closed)
    """
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    ax.plot(lons, lats, linewidth=linewidth, color='tab:red',transform=ccrs.PlateCarree())


def plot_outer_border(ax, geometry: dict, linewidth: float):
    """
    Plot the outer border geometry.
    Supports:
      - LineString
      - LinearRing
      - MultiLineString
      - Polygon (plots exterior only)
      - MultiPolygon (plots each exterior)
    """
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")

    if gtype in ("LineString", "LinearRing"):
        plot_line_like(ax, coords, linewidth)

    elif gtype == "MultiLineString":
        for line in coords:
            plot_line_like(ax, line, linewidth)

    elif gtype == "Polygon":
        # GeoJSON Polygon coords: [exterior_ring, hole1, hole2, ...]
        exterior = coords[0]
        plot_line_like(ax, exterior, linewidth)

    elif gtype == "MultiPolygon":
        for poly in coords:
            exterior = poly[0]
            plot_line_like(ax, exterior, linewidth)

    else:
        raise ValueError(f"Unsupported geometry type: {gtype}")


def add_gridlines(ax):
    """Add labeled gridlines to the map."""
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False


if __name__ == "__main__":
    
    # Read GeoJSON
    gj         = read_geojson(filename_in)
    features   = gj.get("features", [])
    feature    = features[0]
    geometry   = feature.get("geometry", {})
    properties = feature.get("properties", {})

    # Set up map
    fig, ax = setup_cartopy_map(figure_size, map_extent)

    # Plot outer border
    plot_outer_border(ax, geometry, line_width)

    # Add gridlines and title
    add_gridlines(ax)
    ax.set_title(f'nve regine catchment',fontsize=12)

    if write2file:
        fig.savefig(filename_out)
    
    plt.show()

