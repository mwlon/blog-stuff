import numpy as np
from tqdm import tqdm
import time
import matplotlib as mpl
import cv2
from pyarrow import parquet as pq
from pyarrow import Table
from matplotlib import pyplot as plt
from typing import Optional
import os
import math_utils
import itertools
from pcodec import ChunkConfig, standalone

TAU = 2 * np.pi
EPS = -1e-24


def bbox(ptss):
  j0 = np.min(ptss[..., 0], axis=-1).astype(np.int32)
  j1 = np.max(np.ceil(ptss[..., 0]), axis=-1).astype(np.int32)
  i0 = np.min(ptss[..., 1], axis=-1).astype(np.int32)
  i1 = np.max(np.ceil(ptss[..., 1]), axis=-1).astype(np.int32)
  return np.stack(
    [
      i0,
      i1,
      j0,
      j1,
      i1 - i0,
      j1 - j0,
    ],
    axis=-1,
  )


def out_sizes(max_x, max_y, scale):
  area = max_x * max_y
  out_h = int(scale * np.sqrt(max_y / max_x))
  out_w = int(scale * np.sqrt(max_x / max_y))
  return out_h, out_w


def calc_out_xys(xy_pts, *, out_w, out_h, max_x, max_y):
  out_xs = out_w * (xy_pts[:, 0] / max_x)
  out_ys = out_h * (1 - xy_pts[:, 1] / max_y)
  return np.stack([out_xs, out_ys], axis=1)


def fill_value_between(sph_pts, out_pts, *, in_value, out_img):
  out_i0, out_i1, out_j0, out_j1, out_dh, out_dw = bbox(out_pts)
  out_dpts = np.float32(out_pts - np.array([out_j0, out_i0])[None, :])

  mask = np.zeros((out_dh, out_dw), dtype=np.float32)
  out_dpts_i32 = (out_dpts * 256).astype(np.int32)
  cv2.fillConvexPoly(mask, out_dpts_i32, 1.0, shift=8)

  out_img_slice = out_img[out_i0:out_i1, out_j0:out_j1]
  sub = in_value - out_img_slice
  delta = mask[:, :, None].astype(np.uint8) * sub
  out_img_slice += delta


def fill_between(sph_pts, out_pts, *, in_img, out_img):
  h, w, _ = in_img.shape
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
    flags=cv2.INTER_NEAREST,
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
  order = np.argsort(pts[:, 1])  # sort by phi
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


def find_triangle_containing(
  triangle_euc, xyz, max_side_lengths2, recent_indices=None, kdtree=None, k=20
):
  # https://graphallthethings.com/posts/interpolation/
  if recent_indices:
    recent = np.array(recent_indices)
    dets = math_utils.calc_orientation_dets(triangle_euc[recent], xyz)
    mask = (
      (dets[0] * dets[1] >= EPS)
      & (dets[1] * dets[2] >= EPS)
      & (dets[2] * dets[0] >= EPS)
    )
    hits = recent[mask]
    if len(hits):
      return hits[0]
  if kdtree is not None:
    _, candidate_idxs = kdtree.query(xyz, k=k)
    dets = math_utils.calc_orientation_dets(triangle_euc[candidate_idxs], xyz)
    mask = (
      (dets[0] * dets[1] >= EPS)
      & (dets[1] * dets[2] >= EPS)
      & (dets[2] * dets[0] >= EPS)
    )
    hits = candidate_idxs[mask]
    if len(hits):
      return hits[0]
  # For performance, we filter out triangles whose max side length is less than
  # the furthest vertex's distance to the target coordinates.
  # This check also serves the purpose of eliminating triangles on the opposite
  # side of the globe, which would otherwise be false positives.
  vecs = triangle_euc - xyz[None, None, :]
  sumsq = np.sum(vecs * vecs, axis=2)
  plausibly_close = max_side_lengths2 >= EPS + np.max(sumsq, axis=1)
  dets = math_utils.calc_orientation_dets(triangle_euc[plausibly_close], xyz)

  contains = np.zeros(triangle_euc.shape[0], dtype=np.bool_)
  contains[plausibly_close] = (
    (dets[0] * dets[1] >= EPS) & (dets[1] * dets[2] >= EPS) & (dets[2] * dets[0] >= EPS)
  )
  containing_idxs = np.where(contains)[0]
  return containing_idxs[0]


def draw_tissot_ellipse(euc, triangle, out_xy, theta, phi, out_img):
  thetaphi = np.array([theta, phi])
  xyz = math_utils.calc_euc(thetaphi[None, :])[0]
  triples = triangle[None, :]
  dets = math_utils.calc_orientation_dets(euc, triples, xyz)
  props = dets / np.sum(dets)
  center = np.sum(props * out_xy[triangle], axis=0)
  center = tuple(np.round(center).astype(np.int64).tolist())

  _, angles, uv_length, wv_length = math_utils.calc_areas_angles_lengths(euc, triples)
  inv_atlas = math_utils.calc_inv_atlas(angles, uv_length, wv_length)
  tangent_vecs = math_utils.calc_tangent_vecs(out_xy, triples)
  distortion = math_utils.calc_distortion(inv_atlas, tangent_vecs)[0]
  u, sigma, _ = np.linalg.svd(distortion)
  sigma /= 30.0
  rotation = np.arctan2(u[0, 1], u[0, 0]) * 360 / TAU

  cv2.ellipse(
    out_img,
    center,
    (round(sigma[0]), round(sigma[1])),
    rotation,
    0,
    360,
    (0, 100, 40, 255),
    -1,
  )


def plot_map(
  name: str,
  sph_pts: np.ndarray,
  xy_pts: np.ndarray,
  triangles: np.ndarray,
  title: str,
  show: bool = True,
  scale: int = 1024,
  draw_lines: bool = False,
  step: Optional[int] = None,
  source: str | None = None,
  tissot: bool = False,
  draw_lat: int | None = None,
  draw_lng: int | None = None,
  shapefile: str | None = None,
  mapcolor_field: str = "MAPCOLORC",
):
  t = time.time()
  if source is None:
    source = "sources/land_shallow_topo_8192.tif"

  if source.startswith("#"):
    r, g, b = int(source[1:3], 16), int(source[3:5], 16), int(source[5:7], 16)
    in_img = np.full((2, 2, 4), [b, g, r, 255], dtype=np.uint8)
  else:
    in_img = cv2.imread(source)
    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2BGRA)
  sph_pts = np.array(sph_pts)
  xy_pts = np.array(xy_pts)
  xy_pts -= np.min(xy_pts, axis=0)[None, :]
  max_x, max_y = np.max(xy_pts, axis=0)
  out_h, out_w = out_sizes(max_x, max_y, scale)
  out_xys = calc_out_xys(xy_pts, out_w=out_w, out_h=out_h, max_x=max_x, max_y=max_y)
  out_img = np.full([out_h, out_w, 4], 0).astype(in_img.dtype)  # transparent

  for idxs in tqdm(triangles, desc="projecting triangles"):
    sub_pts = calc_sub_pts(sph_pts[idxs], out_xys[idxs])
    for sub_sph, sub_xy in sub_pts:
      fill_between(
        sub_sph,
        sub_xy,
        in_img=in_img,
        out_img=out_img,
      )

  tissot_scale = 24
  tissot_radians = TAU / tissot_scale
  if tissot or shapefile:
    euc = math_utils.calc_euc(sph_pts)
    max_side_lengths2 = math_utils.calc_max_side_lengths2(euc, triangles)

  if shapefile:
    draw_countries(
      name,
      euc,
      triangles,
      max_side_lengths2,
      out_xys,
      out_img,
      shapefile_path=shapefile,
      mapcolor_field=mapcolor_field,
    )

  if tissot:
    triangle_euc = np.take(euc, triangles, axis=0)
    for i in range(1, tissot_scale):
      for j in range(1, tissot_scale // 2):
        xyz = math_utils.calc_euc(np.array([[theta, phi]]))[0]
        triangle_idx = find_triangle_containing(
          triangle_euc, xyz, phi, max_side_lengths2
        )
        triangle = triangles[triangle_idx]

        draw_tissot_ellipse(euc, triangle, out_xys, theta, phi, out_img)

  if draw_lines:
    for idxs in triangles:
      sub_pts = calc_sub_pts(sph_pts[idxs], out_xys[idxs])
      color = [255, 255, 255, 255] if len(sub_pts) == 1 else [0, 200, 200, 255]
      # color = [0, 0, 255, 255] if len(sub_pts) == 1 else [0, 200, 200, 255]
      for _, sub_xy in sub_pts:
        sub_xy = sub_xy.astype(np.int64)
        for j, k in [[0, 1], [1, 2], [2, 0]]:
          cv2.line(out_img, sub_xy[j], sub_xy[k], color=color)

  theta, phi = np.ascontiguousarray(sph_pts.transpose())
  out_xys_int = out_xys.astype(np.int64)
  if draw_lat:
    for lat_deg in range(draw_lat, 180, draw_lat):
      target_phi = lat_deg * TAU / 360
      # adjust to a phi that actually exists
      target_phi = phi[np.argmin(np.abs(phi - target_phi))]
      mask = np.abs(phi - target_phi) < 1e-6
      thetas = theta[mask]
      idxs = np.where(mask)[0][np.argsort(thetas)]
      for i, j in itertools.pairwise(idxs):
        cv2.line(
          out_img,
          out_xys_int[i],
          out_xys_int[j],
          color=[255, 255, 255, 255],
          thickness=2,
        )
  if draw_lng:
    polar_theta = theta[phi < (15.0 / 360 * TAU)]
    for lng_deg in range(draw_lng, 360, draw_lng):
      target_theta = lng_deg * TAU / 360
      # adjust to a theta that actually exists near the poles
      target_theta = polar_theta[np.argmin(np.abs(polar_theta - target_theta))]
      mask = np.abs(theta - target_theta) < 1e-6
      phis = phi[mask]
      idxs = np.where(mask)[0][np.argsort(phis)]
      for i, j in itertools.pairwise(idxs):
        cv2.line(
          out_img,
          out_xys_int[i],
          out_xys_int[j],
          color=[255, 255, 255, 255],
          thickness=2,
        )
  print("drew map in", time.time() - t)

  fname = f"{title}_{step:05d}.png" if step is not None else f"{title}.png"
  dir_ = f"results/{name}"
  os.makedirs(dir_, exist_ok=True)
  success = cv2.imwrite(f"{dir_}/{fname}", out_img)
  if not success:
    raise Exception("failed to save")
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
  if "area" in label:
    norm = mpl.colors.LogNorm(vmin=0.01, vmax=100)
    cmap = mpl.cm.get_cmap("bwr")
  else:
    norm = mpl.colors.LogNorm(vmin=1, vmax=100)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("wr", ["#ffffff", "#ff0000"])

  ax.set_aspect("equal")
  ax.axis("off")
  ax.text(label_x, label_y, label, va="center", ha="center")
  ax.set_xlim(np.min(xy_triangles[:, :, 0]), np.max(xy_triangles[:, :, 0]))
  ax.set_ylim(np.min(xy_triangles[:, :, 1]), np.max(xy_triangles[:, :, 1]))
  # we can't use a matplotlib triangulation because our loss (color) is per
  # triangle rather than per point
  for i in range(k):
    ax.add_patch(
      plt.Polygon(
        xy_triangles[i],
        color=cmap(norm(mean_mults[i])),
      )
    )
  plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)


def plot_mults(
  name,
  xy,  # n x 2
  triangles,  # k x 3
  area_mults,  # 3k
  angle_mults,  # 3k
  show: bool = True,
  title: Optional[str] = None,
):
  # https://matplotlib.org/stable/gallery/images_contours_and_fields/triinterp_demo.html#sphx-glr-gallery-images-contours-and-fields-triinterp-demo-py
  mpl.rc("font", family="sans-serif", size=15)
  plt.tight_layout()
  # k x 3 x 2
  xy_triangles = np.take_along_axis(xy[None, :, :], triangles[:, :, None], axis=1)
  fig, (ax0, ax1) = plt.subplots(2, figsize=(9, 10))
  plot_single_mults(xy_triangles, area_mults, ax0, "areal")
  plot_single_mults(xy_triangles, angle_mults, ax1, "angular")
  if title is not None:
    fig.suptitle(title, fontsize=16)
  plt.savefig(f"results/{name}/mults.png")
  if show:
    plt.show()


def _load_or_compute_indices(
  name, shapefile_path, triangle_euc, triangles, max_side_lengths2
):
  stem = os.path.splitext(os.path.basename(shapefile_path))[0]
  cache_path = f"results/{name}/{stem}_indices.pco"
  if os.path.exists(cache_path):
    return standalone.simple_decompress(open(cache_path, "rb").read()).astype(np.int32)
  import shapefile as pyshp

  sf = pyshp.Reader(shapefile_path)
  all_points = np.array([p for s in sf.iterShapes() for p in s.points])
  theta = (all_points[:, 0] + 180) / 360 * TAU
  phi = (90 - all_points[:, 1]) / 180 * (TAU / 2)
  xyz = math_utils.calc_euc(np.stack([theta, phi], axis=1))

  from scipy.spatial import KDTree

  kdtree = KDTree(triangle_euc.mean(axis=1))

  def compute_indices(xyz, n_recent=3):
    recent = []
    for x in tqdm(xyz, desc="building shapefile index cache"):
      idx = find_triangle_containing(
        triangle_euc, x, max_side_lengths2, recent_indices=recent, kdtree=kdtree
      )
      if idx not in recent:
        if len(recent) == n_recent:
          recent.pop(0)
        recent.append(idx)
      yield idx

  indices = np.fromiter(compute_indices(xyz), dtype=np.int32, count=len(xyz))
  with open(cache_path, "wb") as f:
    f.write(standalone.simple_compress(indices, ChunkConfig()))
  return indices


DEFAULT_MAPCOLORS = [
  [80, 110, 170, 255],  # orange
  [80, 140, 55, 255],  # green
  [180, 115, 65, 255],  # turquoise
  [110, 80, 160, 255],  # red
  [190, 80, 140, 255],  # purple
  [70, 145, 135, 255],  # yellow
  [255, 255, 255, 255],  # white
]


def draw_countries(
  name,
  euc,
  triangles,
  max_side_lengths2,
  out_xys,
  out_img,
  shapefile_path,
  mapcolor_field,
):
  """Draw filled countries using mapcolor_field to index fill_colors."""
  import shapefile as pyshp

  fill_colors = DEFAULT_MAPCOLORS

  triangle_euc = np.take(euc, triangles, axis=0)
  indices = _load_or_compute_indices(
    name, shapefile_path, triangle_euc, triangles, max_side_lengths2
  )

  # Project all shapefile points at once
  sf = pyshp.Reader(shapefile_path)
  all_lonlat = np.array([p for s in sf.iterShapes() for p in s.points])
  all_theta = (all_lonlat[:, 0] + 180) / 360 * TAU
  all_phi = (90 - all_lonlat[:, 1]) / 180 * (TAU / 2)
  all_xyz = math_utils.calc_euc(np.stack([all_theta, all_phi], axis=1))
  dets = math_utils.calc_orientation_dets(triangle_euc[indices], all_xyz)
  weights = np.stack(dets, axis=1)  # [total_pts, 3]
  weights /= weights.sum(axis=1, keepdims=True)
  all_projected = np.einsum("ni,nij->nj", weights, out_xys[triangles[indices]])

  pt_offset = 0
  for sr in tqdm(sf.iterShapeRecords(), desc="drawing shapes", total=len(sf)):
    fill_color = fill_colors[sr.record[mapcolor_field] - 1]
    shape = sr.shape
    n_pts = len(shape.points)
    parts = list(shape.parts) + [n_pts]
    for i in range(len(shape.parts)):
      pts = all_projected[pt_offset + parts[i] : pt_offset + parts[i + 1]]
      if len(pts) < 3:
        continue
      pts_int = np.round(pts).astype(np.int32)
      cv2.fillPoly(out_img, [pts_int], color=fill_color)
    pt_offset += n_pts


def detect_water(source):
  earth = cv2.imread(source)
  b, g, r = earth.transpose([2, 0, 1])
  water_color = (b > 45) & (g < 50) & (r < 50)
  plausible_position = np.ones_like(water_color)
  # parts of antarctica look like water, so we hack some of them to be land
  h, w, _ = earth.shape
  antarctica_land_h = int((h * 260) / 4096)
  plausible_position[-antarctica_land_h:] = 0
  return water_color & plausible_position


def calc_water_prop(sph, triangles, source):
  res = np.zeros(triangles.shape[0], dtype=np.float32)
  is_water = detect_water(source)
  h, w = is_water.shape
  sphs = np.take(sph, triangles, axis=0)
  in_xss = sphs[:, :, 0] / TAU * w
  in_yss = sphs[:, :, 1] / (TAU / 2) * h

  in_ptss = np.stack([in_xss, in_yss], axis=-1)
  n_triangles = triangles.shape[0]
  in_i01j01dhdw = bbox(in_ptss)

  for i in range(n_triangles):
    in_pts = in_ptss[i]
    in_i0, in_i1, in_j0, in_j1, in_dh, in_dw = in_i01j01dhdw[i]
    sub_image = is_water[in_i0:in_i1, in_j0:in_j1]
    mask = np.zeros(sub_image.shape, dtype=np.float32)
    in_dpts = (in_pts - np.array([in_j0, in_i0])[None, :]).astype(np.int32)
    cv2.fillConvexPoly(mask, in_dpts, color=1.0)
    res[i] = np.sum(mask * sub_image) / np.sum(mask)

  return res
