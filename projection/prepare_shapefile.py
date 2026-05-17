"""
Prepares a shapefile for use in map rendering:
  1. Nudges all longitudes within EPSILON of ±180 inward, preventing
     antimeridian-crossing polygons from drawing spurious lines.
  2. Derives a MAPCOLORC (custom) field from the Natural Earth MAPCOLOR7 field:
       - Spain and territories: 4→6
       - France and territories: 7→4
       - Antarctica: →7 (reserved for the ice sheet)
     Colors 1–6 are used for all other countries; 7 is Antarctica only.
  3. Derives a MAPCOLORD field like MAPCOLORC but fixes same-color adjacent pairs
     by recoloring sovereigns so territories can inherit their color:
       - United Kingdom and all territories: 6→5
       - Netherlands and all territories: 4→5
       - Palestine: 3→1 (differs from Israel)
       - USNB Guantanamo Bay: 3→1 (differs from Cuba)

Usage: python prepare_shapefile.py input.shp output.shp
"""
import sys
import shapefile

EPSILON = 0.01  # degrees, for antimeridian nudging


def nudge(lon):
    return max(-(180 - EPSILON), min(180 - EPSILON, lon))


def remap_color(record):
    c = record["MAPCOLOR7"]
    sov = record["SOVEREIGNT"]
    if sov == "Antarctica":
        return 7
    if sov == "Spain":
        return 6
    if sov == "France":
        return 4
    return c


# Individual name overrides applied after sovereign remapping
COLOR_D_NAME_OVERRIDES = {
    "Palestine": 1,
    "USNB Guantanamo Bay": 4,
    "Akrotiri": 2,
    "Dhekelia": 2,
}

# Sovereign-level remapping for MAPCOLORD
COLOR_D_SOV_OVERRIDES = {
    "United Kingdom": 1,
    "Netherlands": 5,
}

COLOR_D_NAME_OVERRIDES_POST_SOV = {
    "Iceland": 6,
}


def remap_color_d(record):
    name = record["NAME"]
    if name in COLOR_D_NAME_OVERRIDES:
        return COLOR_D_NAME_OVERRIDES[name]
    sov = record["SOVEREIGNT"]
    if sov in COLOR_D_SOV_OVERRIDES:
        return COLOR_D_SOV_OVERRIDES[sov]
    if name in COLOR_D_NAME_OVERRIDES_POST_SOV:
        return COLOR_D_NAME_OVERRIDES_POST_SOV[name]
    return remap_color(record)


def main(src_path, dst_path):
    sf = shapefile.Reader(src_path)
    print(f"Read {len(sf)} shapes")

    with shapefile.Writer(dst_path) as w:
        w.fields = sf.fields[1:]  # skip deletion flag
        w.field("MAPCOLORC", "N", 3, 0)
        w.field("MAPCOLORD", "N", 3, 0)
        for sr in sf.iterShapeRecords():
            nudged_points = [(nudge(lon), lat) for lon, lat in sr.shape.points]
            new_shape = shapefile.Shape(
                shapeType=sr.shape.shapeType,
                points=nudged_points,
                parts=sr.shape.parts,
            )
            w.record(*sr.record, remap_color(sr.record), remap_color_d(sr.record))
            w.shape(new_shape)

    print(f"Wrote {dst_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.shp output.shp")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
