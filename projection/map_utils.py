import numpy as np
import matplotlib as mpl
import cv2
from matplotlib import pyplot as plt
from datetime import datetime
from matplotlib import tri
from typing import Optional

TAU = 2 * np.pi
in_img = cv2.imread('land_shallow_topo_8192.tif')
h, w, _ = in_img.shape

def bbox(pts):
  j0 = int(np.min(pts[:, 0]))
  j1 = int(np.max(np.ceil(pts[:, 0])))
  i0 = int(np.min(pts[:, 1]))
  i1 = int(np.max(np.ceil(pts[:, 1])))
  return i0, i1, j0, j1, i1 - i0, j1 - j0

def out_sizes(max_x, max_y, scale):
  area = max_x * max_y
  out_h = int(scale * np.sqrt(max_y / max_x))
  out_w = int(scale * np.sqrt(max_x / max_y))
  return out_h, out_w

def calc_out_xys(xy_pts, *, out_w, out_h, max_x, max_y):
  out_xs = out_w * (xy_pts[:, 0] / max_x)
  out_ys = out_h * (1 - xy_pts[:, 1] / max_y)
  return np.stack([out_xs, out_ys], axis=1)

def fill_between(sph_pts, out_pts, *, out_img):
  in_xs = sph_pts[:, 0] / TAU * w
  in_ys = sph_pts[:, 1] / (TAU / 2) * h
  in_pts = np.stack([in_xs, in_ys], axis=1)

  out_i0, out_i1, out_j0, out_j1, out_dh, out_dw = bbox(out_pts)
  out_dpts = np.float32(out_pts - np.array([out_j0, out_i0])[None, :])

  in_i0, in_i1, in_j0, in_j1, in_dh, in_dw = bbox(in_pts)
  in_dpts = np.float32(in_pts - np.array([in_j0, in_i0])[None, :])

  warp_mat = cv2.getAffineTransform(in_dpts, out_dpts)
  warped = cv2.warpAffine(
   in_img[in_i0:in_i1, in_j0:in_j1],
   warp_mat, 
   (out_dw, out_dh), 
   borderMode=cv2.BORDER_REFLECT_101,
  )

  mask = np.zeros((out_dh, out_dw), dtype=np.float32)
  out_dpts_i32 = (out_dpts * 256).astype(np.int32)
  cv2.fillConvexPoly(mask, out_dpts_i32, 1.0, shift=8)
  
  out_img_slice = out_img[out_i0:out_i1, out_j0:out_j1]
  sub = warped - out_img_slice
  delta = mask[:, :, None].astype(np.uint8) * sub
  out_img_slice += delta

def calc_sub_pts(sph, out_xy):
  pts = np.concatenate([sph, out_xy], axis=1)
  order = np.argsort(pts[:, 1]) # sort by phi
  pts = pts[order]
  min_phi, mid_phi, max_phi = pts[:, 1]
  # this is a hack to add a bit more resolution near the poles
  # it turns triangles with a vertex at a pole into 3
  if min_phi == 0 and mid_phi > 0:
    cutoff = mid_phi / 2
    w01 = cutoff / max(mid_phi, cutoff)
    w02 = cutoff / max_phi
    pt01 = w01 * pts[1] + (1 - w01) * pts[0]
    pt01[0] = pts[1, 0]
    pt02 = w02 * pts[2] + (1 - w02) * pts[0]
    pt02[0] = pts[2, 0]
    sub_pts = [
      [pts[0], pt01, pt02],
      [pts[1], pt01, pts[2]],
      [pts[2], pt02, pt01],
    ]
  elif max_phi == TAU / 2 and mid_phi < TAU / 2:
    cutoff = (TAU / 2 - mid_phi) / 2
    w02 = cutoff / max(TAU / 2 - mid_phi, cutoff)
    w12 = cutoff / (TAU / 2 - min_phi)
    pt02 = w02 * pts[0] + (1 - w02) * pts[2]
    pt02[0] = pts[0, 0]
    pt12 = w12 * pts[1] + (1 - w12) * pts[2]
    pt12[0] = pts[1, 0]
    sub_pts = [
      [pts[0], pts[1], pt02],
      [pts[1], pt02, pt12],
      [pts[2], pt02, pt12],
    ]
  else:
    sub_pts = [pts]

  return [(np.array(x)[:, :2], np.array(x)[:, 2:]) for x in sub_pts]

def plot_map(
  name: str,
  sph_pts: np.ndarray,
  xy_pts: np.ndarray,
  triangles: np.ndarray,
  show: bool = True,
  scale: int = 1024,
  draw_lines: bool = False,
  step: Optional[int] = None,
):
  sph_pts = np.array(sph_pts)
  xy_pts = np.array(xy_pts)
  xy_pts -= np.min(xy_pts, axis=0)[None, :]
  n = sph_pts.shape[0]
  max_x, max_y = np.max(xy_pts, axis=0)
  out_h, out_w = out_sizes(max_x, max_y, scale)
  out_xys = calc_out_xys(xy_pts, out_w=out_w, out_h=out_h, max_x=max_x, max_y=max_y)
  out_img = np.full([out_h, out_w, 3], 255).astype(in_img.dtype)
  #out_img[:, :, 0] = 0
  #out_img[:, :, 1] = 0

  for idxs in triangles:
    sub_pts = calc_sub_pts(sph_pts[idxs], out_xys[idxs])
    for sub_sph, sub_xy in sub_pts:
      fill_between(
        sub_sph,
        sub_xy,
        out_img=out_img,
      )

  if draw_lines:
    for idxs in triangles:
      sub_pts = calc_sub_pts(sph_pts[idxs], out_xys[idxs])
      color = [0, 0, 255] if len(sub_pts) == 1 else [0, 200, 200]
      for _, sub_xy in sub_pts:
        sub_xy = sub_xy.astype(np.int64)
        for j, k in [[0, 1], [1, 2], [2, 0]]:
          cv2.line(out_img, sub_xy[j], sub_xy[k], color=color)
  
  fname = f'earth_{step:05d}.png' if step is not None else 'earth.png'
  success = cv2.imwrite(f'results/{name}/{fname}', out_img)
  if not success:
    raise Exception('failed to save')
  if show:
    cv2.imshow(name, out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def shape_mults(mults):
  return mults.reshape([3, -1]).transpose()

def plot_single_mults(xy_triangles, mults, ax, label):
  k = xy_triangles.shape[0]
  shaped_mults = shape_mults(mults)
  mean_mults = np.mean(shaped_mults, axis=1)

  label_x = np.min(xy_triangles[:, :, 0] - 0.5)
  label_y = np.mean(xy_triangles[:, :, 1])
  if 'area' in label:
    norm = mpl.colors.LogNorm(vmin=0.01, vmax=100)
    cmap = mpl.cm.get_cmap('bwr')
  else:
    norm = mpl.colors.LogNorm(vmin=1, vmax=100)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('wr', ['#ffffff', '#ff0000'])

  ax.set_aspect('equal')
  ax.axis("off")
  ax.text(label_x, label_y, label, va='center', ha='center')
  ax.set_xlim(np.min(xy_triangles[:, :, 0]), np.max(xy_triangles[:, :, 0]))
  ax.set_ylim(np.min(xy_triangles[:, :, 1]), np.max(xy_triangles[:, :, 1]))
  # we can't use a matplotlib triangulation because our loss (color) is per
  # triangle rather than per point
  for i in range(k):
    ax.add_patch(plt.Polygon(
      xy_triangles[i],
      color=cmap(norm(mean_mults[i])),
    ))
  plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

def plot_mults(
  name,
  xy, # n x 2
  triangles, # k x 3
  area_mults, # 3k
  angle_mults, # 3k
):
  #https://matplotlib.org/stable/gallery/images_contours_and_fields/triinterp_demo.html#sphx-glr-gallery-images-contours-and-fields-triinterp-demo-py
  mpl.rc('font', family='sans-serif', size=15)
  plt.tight_layout()
  k = triangles.shape[0]
  # k x 3 x 2
  xy_triangles = np.take_along_axis(xy[None, :, :], triangles[:, :, None], axis=1)
  fig, (ax0, ax1) = plt.subplots(2, figsize=(9, 10))
  plot_single_mults(xy_triangles, area_mults, ax0, 'areal')
  plot_single_mults(xy_triangles, angle_mults, ax1, 'angular')
  plt.savefig(f'results/{name}/mults.png')
  plt.show()
