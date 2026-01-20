"""
Create a GeoJSON containing the DISSOLVED CATCHMENT POLYGON (Polygon/MultiPolygon)
from many NVE REGINE subcatchment polygons.

Input was downloaded here: https://nedlasting.nve.no/gis/ using
kartformat = geojson, koordinatsystem = WGS84 bredde/lengdegrader,
utvalgsmetode = helt innenfor, dekningsomrode = vassdgragsomrode

Output should be a Polygon/MultiPolygon (filled area), which is suitable for:
- plotting (you can derive boundary later)
- area weights / intersections
- masking / clipping

Workflow:
1) Read the input GeoJSON (many REGINE features).
2) Convert all feature geometries to Shapely polygons.
3) Dissolve (union) all polygons into one catchment geometry.
4) Write a new GeoJSON FeatureCollection containing that dissolved geometry.
"""

import json
from pathlib               import Path
from shapely.geometry      import Polygon, mapping
from shapely.ops           import unary_union
from Dunnsigouin_etal_2026 import config


# input --------------------------------------------------------------
path_in      = config.dirs["nve_catchment"]
path_out     = config.dirs["nve_catchment"]
filename_in  = Path(f"{path_in}nve_regine_enhet_012_drammensvassdraget_subcatchments.geojson")
filename_out = Path(f"{path_out}nve_regine_enhet_012_drammensvassdraget_entire_catchment.geojson")
ignore_holes = True   # If True, ignores holes in each subcatchment polygon
write2file   = True
# --------------------------------------------------------------------


def read_geojson(filepath: Path) -> dict:
    """Read a GeoJSON file into a Python dict."""
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def features_to_polygons(features: list, ignore_holes: bool = True) -> list:
    """
    Convert GeoJSON features to a list of Shapely Polygons.

    Supports:
      - Polygon
      - MultiPolygon

    Notes:
      - If ignore_holes=True, only the exterior ring is used (coordinates[0]).
      - If ignore_holes=False, interior rings are passed as 'holes' to Polygon().
    """
    polygons = []

    for feat in features:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates")

        if not coords:
            continue

        if gtype == "Polygon":
            exterior = coords[0]
            holes = None if ignore_holes or len(coords) <= 1 else coords[1:]
            polygons.append(Polygon(exterior, holes=holes))

        elif gtype == "MultiPolygon":
            # coords: [poly1, poly2, ...] where each poly is [rings...]
            for poly in coords:
                exterior = poly[0]
                holes = None if ignore_holes or len(poly) <= 1 else poly[1:]
                polygons.append(Polygon(exterior, holes=holes))

        else:
            # Skip unexpected geometry types (e.g., Point/LineString)
            continue

    return polygons


def dissolve_polygons(polygons: list):
    """Dissolve a list of polygons into a single geometry (Polygon or MultiPolygon)."""
    if not polygons:
        raise ValueError("No polygons found to dissolve.")
    return unary_union(polygons)


def dissolved_polygon_geojson_geometry(dissolved_geom) -> dict:
    """
    Return the dissolved geometry as a GeoJSON geometry dict.
    Expected types:
      - Polygon
      - MultiPolygon
    """
    gtype = dissolved_geom.geom_type
    if gtype not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Unexpected dissolved geometry type: {gtype}")
    return mapping(dissolved_geom)


def write_catchment_geojson(out_path: Path, geom_dict: dict, properties: dict, write2file: bool):
    """Write a FeatureCollection containing one Feature: the dissolved catchment polygon."""
    out_gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geom_dict,
                "properties": properties or {},
            }
        ],
    }

    if write2file:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_gj, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    
    gj = read_geojson(filename_in)
    features = gj.get("features", [])
    print(f"Read {len(features)} features from: {filename_in.name}")

    polygons = features_to_polygons(features, ignore_holes=ignore_holes)
    print(f"Converted to {len(polygons)} polygons")

    dissolved = dissolve_polygons(polygons)
    print(f"Dissolved geometry type: {dissolved.geom_type}")

    catchment_geom = dissolved_polygon_geojson_geometry(dissolved)

    props = {
        "source": "NVE REGINE dissolved catchment polygon",
        "input_file": filename_in.name,
        "ignore_holes": bool(ignore_holes),
        "note": "Dissolved Polygon/MultiPolygon (use .boundary later if you need an outline)",
    }

    write_catchment_geojson(filename_out, catchment_geom, props, write2file)
    print(f"Wrote catchment polygon GeoJSON: {filename_out}")
