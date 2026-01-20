"""
Create a GeoJSON containing ONLY the outer border of an NVE REGINE catchment
based on the smaller subcatchments within. 

Input was downloaded here: https://nedlasting.nve.no/gis/ using
kartformat = geojson, koordinatsystem = WGS84 bredde/lengdegrader,
utvalgsmetode = helt innenfor, dekningsomrode = vassdgragsomrode

Output should look like the largest regine catchments found here:
https://temakart.nve.no/tema/nedborfelt

Workflow:
1) Read the input GeoJSON (many REGINE features).
2) Convert all feature geometries to Shapely polygons.
3) Dissolve (union) all polygons into one catchment geometry.
4) Extract the outer boundary (LineString / MultiLineString).
5) Write a new GeoJSON FeatureCollection containing only that border geometry.
"""

import json
from pathlib               import Path
from shapely.geometry      import Polygon, mapping
from shapely.ops           import unary_union
from Dunnsigouin_etal_2026 import config


# user input parameters ------------------------------------------------------
path_in      = config.dirs["nve_catchment"]
path_out     = config.dirs["nve_catchment"]
filename_in  = Path(f'{path_in}nve_regine_enhet_012_drammensvassdraget_subcatchments.geojson')
filename_out = Path(f'{path_in}nve_regine_enhet_012_drammensvassdraget_entire_catchment.geojson')
ignore_holes = True # If True, holes are ignored when building polygons (usually fine for catchment outline)
write2file   = True
# ----------------------------------------------------------------------------


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
    """
    Dissolve a list of polygons into a single geometry (Polygon or MultiPolygon).
    """
    if not polygons:
        raise ValueError("No polygons found to dissolve.")
    return unary_union(polygons)


def outer_border_geojson_geometry(dissolved_geom: object) -> dict:
    """
    Extract ONLY the outer boundary of a dissolved Polygon/MultiPolygon
    and return it as a GeoJSON geometry dict.

    Output types:
      - LineString (if a single Polygon)
      - MultiLineString (if a MultiPolygon)
    """
    gtype = dissolved_geom.geom_type

    if gtype == "Polygon":
        # exterior is a LineString
        return mapping(dissolved_geom.exterior)

    if gtype == "MultiPolygon":
        # build a MultiLineString from each polygon exterior
        coords = [mapping(poly.exterior)["coordinates"] for poly in dissolved_geom.geoms]
        return {"type": "MultiLineString", "coordinates": coords}

    raise ValueError(f"Unexpected dissolved geometry type: {gtype}")


def write_outerborder_geojson(out_path, border_geom_dict, properties, write2file):
    """
    Write a FeatureCollection containing one Feature: the outer border geometry.
    """
    out_gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": border_geom_dict,
                "properties": properties or {},
            }
        ],
    }
    if write2file:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_gj, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    
    gj       = read_geojson(filename_in)
    features = gj.get("features", [])
    print(f"Read {len(features)} features from: {filename_in.name}")

    polygons = features_to_polygons(features, ignore_holes=ignore_holes)
    print(f"Converted to {len(polygons)} polygons")

    dissolved = dissolve_polygons(polygons)
    print(f"Dissolved geometry type: {dissolved.geom_type}")

    border_geom = outer_border_geojson_geometry(dissolved)

    props = {
        "source": "NVE REGINE union outer border",
        "input_file": filename_in.name,
        "ignore_holes": bool(ignore_holes),
    }

    write_outerborder_geojson(filename_out, border_geom, props, write2file)
    print(f"Wrote outer border GeoJSON: {filename_out}")




