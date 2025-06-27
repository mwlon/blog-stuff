import cv2
import numpy as np
import map_utils
import argparse
import lattice

parser = argparse.ArgumentParser()
parser.add_argument(
  '--side-n',
  type=int,
  default=None,
  help='number of lattice points around equator, if you want to draw triangles',
)
parser.add_argument(
  '--draw-lines',
  action='store_true',
)
args = parser.parse_args()

is_water = map_utils.detect_water()
if args.side_n is None:
  out_img = np.where(is_water, 255, 0).astype(np.uint8)
else:
  lattice = lattice.build_lattice(args.side_n)
  sph = lattice.sph
  triangles = lattice.triangles
  xy = lattice.sph.copy()
  water_prop = map_utils.calc_water_prop(sph, triangles)
  water_prop_uint8 = (water_prop * 255.0).astype(np.uint8)

  xy[:, 1] *= -1
  xy -= np.min(xy, axis=0)[None, :]
  n = sph.shape[0]
  max_x, max_y = np.max(xy, axis=0)
  out_h, out_w = is_water.shape
  out_xys = map_utils.calc_out_xys(xy, out_w=out_w, out_h=out_h, max_x=max_x, max_y=max_y)
  out_img = np.full([out_h, out_w, 3], 255).astype(np.uint8)

  for i, idxs in enumerate(triangles):
    sub_pts = map_utils.calc_sub_pts(sph[idxs], out_xys[idxs])
    for sub_sph, sub_xy in sub_pts:
      map_utils.fill_value_between(
        sub_sph,
        sub_xy,
        in_value=water_prop_uint8[i],
        out_img=out_img,
      )

  if args.draw_lines:
    for idxs in triangles:
      sub_pts = calc_sub_pts(sph[idxs], out_xys[idxs])
      color = [0, 0, 255] if len(sub_pts) == 1 else [0, 200, 200]
      for _, sub_xy in sub_pts:
        sub_xy = sub_xy.astype(np.int64)
        for j, k in [[0, 1], [1, 2], [2, 0]]:
          cv2.line(out_img, sub_xy[j], sub_xy[k], color=color)

print(out_img.shape, out_img.dtype)
cv2.imwrite('water.png', out_img)
cv2.imshow('water', out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
