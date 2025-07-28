import cv2
import numpy as np
from argparse import ArgumentParser
import map_utils

parser = ArgumentParser()
parser.add_argument(
  "--source",
  type=str,
  help="image to rotate about axis",
)
parser.add_argument(
  "--dest",
  type=str,
  help="where to write the output",
)
parser.add_argument(
  "--detect-water",
  action="store_true",
)
parser.add_argument(
  "--water-hsv",
  type=str,
  default="180,0,255",
  help="comma-separated HSV values for detected water",
)
parser.add_argument(
  "--land-hsv",
  type=str,
  default="180,0,0",
  help="comma-separated HSV values for detected land",
)
parser.add_argument(
  "--rotation",
  type=float,
  default=0.0,
  help="degrees to rotate by",
)
parser.add_argument(
  "--sea-green",
  action="store_true",
  help="make the seas greener",
)
parser.add_argument(
  "--bathymetry",
  type=str,
  help="path to bathymetry image",
)
args = parser.parse_args()


def parse_hsv(hsv_str):
  items = hsv_str.split(",")
  return np.array([float(x) for x in items])


if args.detect_water or args.bathymetry:
  is_water = map_utils.detect_water(args.source)

if args.detect_water:
  is_water_f32 = is_water.astype(np.float32)
  out_hsv = (is_water_f32[:, :, None] * parse_hsv(args.water_hsv)[None, None, :]) + (
    (1.0 - is_water_f32[:, :, None]) * parse_hsv(args.land_hsv)[None, None, :]
  )
  out = cv2.cvtColor(out_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
else:
  img = cv2.imread(args.source)
  out = img.copy()

h, w, _ = out.shape
if args.rotation:
  rot_x = int(w * args.rotation / 360.0)
  out[:, :rot_x] = img[:, -rot_x:]
  out[:, rot_x:] = img[:, :-rot_x]

if args.sea_green:
  tmp = out.astype(np.float32)
  src_sea_blue = 50.0
  b_ceil = np.minimum(tmp[:, :, 0], src_sea_blue)
  g = tmp[:, :, 1]
  r = tmp[:, :, 2]
  prop = b_ceil / src_sea_blue * (1.0 - g / 255.0) ** 2 * (1 - r / 255.0) ** 2
  tmp[:, :, 1] = np.maximum(g, prop * b_ceil)
  tmp[:, :, 2] = np.maximum(0, tmp[:, :, 2] - 5.0 * prop)
  tmp *= (prop * 0.9 + (1 - prop))[:, :, None]
  out = tmp.astype(np.uint8)

if args.bathymetry:
  tmp = out.astype(np.float32)
  bathy = cv2.imread(args.bathymetry)
  assert bathy.shape == tmp.shape
  assert bathy.shape[2] == 3  # bgr
  shallowness = np.sum(bathy, axis=2)
  is_water = is_water | (shallowness < 765)
  incr = (shallowness * is_water) / 765.0
  tmp[..., 0] = tmp[..., 0] - 10 + incr * 25
  tmp[..., 1] = tmp[..., 1] + 13 + incr * 30
  tmp[..., 2] = tmp[..., 2] - 7 + incr * 6.0
  tmp = np.clip(tmp, 0, 255).astype(np.uint8)
  out = (~is_water[:, :, None]) * out + is_water[:, :, None] * tmp

cv2.imwrite(args.dest, out)
