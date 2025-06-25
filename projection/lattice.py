from dataclasses import dataclass
import numpy as np
from typing import List

TAU = 2 * np.pi

@dataclass
class Lattice:
  sph: np.array # n x 2
  euc: np.array # n x 3
  triangles: np.array # k x 3, int

  def triples(self):
    # 3k x 3, int
    return triples_for_triangles(self.triangles)

  def triple_triangle_idxs(self):
    return np.tile(np.arange(self.triangles.shape[0]), 3)
    

@dataclass
class RowSpec:
  base_idx: int
  n: int

def check_ratio_compatible(n0, n1):
  lo = min(n0, n1)
  hi = max(n0, n1)
  if lo > 1:
    r = (hi - 1) // (lo - 1)
    # check if it's not a power of 2
    if r & (r - 1) != 0:
      raise ValueError('unable to build grid with non power of 2 downscaling')

def small_ratio_idxs(n, r):
  return (0.5 + r * np.arange(n - 1)).astype(np.int64)

def big_ratio_idxs(n, r):
  return (r / 2 + r * np.arange(n - 1)).astype(np.int64)

def safe_div(n0, n1):
  if n1 == 1:
    return 10000
  return (n0 - 1) / (n1 - 1)

def triangles_for_rows(row_0: RowSpec, row_1: RowSpec, toggle_left: bool):
  check_ratio_compatible(row_0.n, row_1.n)
  ratio_0_to_1 = safe_div(row_0.n, row_1.n)
  ratio_1_to_0 = safe_div(row_1.n, row_0.n)

  if ratio_0_to_1 < 1:
    off_0 = small_ratio_idxs(row_1.n, ratio_0_to_1)
    off_1 = big_ratio_idxs(row_0.n, ratio_1_to_0)
  elif ratio_0_to_1 == 1:
    n = row_0.n
    mask = np.ones(n - 1, dtype=np.int64)
    if toggle_left:
      mask[:n // 2] = 0
    else:
      mask[n // 2:] = 0
    off_0 = np.arange(n - 1) + mask
    off_1 = np.arange(n - 1) + (1 - mask)
  else:
    off_0 = big_ratio_idxs(row_1.n, ratio_0_to_1)
    off_1 = small_ratio_idxs(row_0.n, ratio_1_to_0)
    
  triples_0_to_1 = np.stack([
    row_0.base_idx + np.arange(row_0.n - 1),
    row_1.base_idx + off_1,
    row_0.base_idx + np.arange(1, row_0.n),
  ], axis=1)
  triples_1_to_0 = np.stack([
    row_1.base_idx + np.arange(row_1.n - 1),
    row_0.base_idx + off_0,
    row_1.base_idx + np.arange(1, row_1.n),
  ], axis=1)
  return np.concatenate([triples_0_to_1, triples_1_to_0], axis=0)

def triangles_for_multi_rows(rows_0: List[RowSpec], rows_1: List[RowSpec], toggle_left: bool):
  triangles = []
  i0, i1, j0, j1 = 0, 0, 0, 0
  #SPAMGETTI
  while j0 < len(rows_0) and j1 < len(rows_1):
    row0 = rows_0[j0]
    row1 = rows_1[j1]
    step = min(row0.n - i0, row1.n - i1)
    triangles.extend(triangles_for_rows(
      RowSpec(row0.base_idx + i0, step),
      RowSpec(row1.base_idx + i1, step),
      toggle_left=toggle_left,
    ))
    
    i0 += step - 1
    i1 += step - 1
    if i0 == row0.n - 1:
      i0 = 0
      j0 += 1
    if i1 == row1.n - 1:
      i1 = 0
      j1 += 1

  assert j0 == len(rows_0) and j1 == len(rows_1)
  return triangles

def triples_for_triangles(triangles):
  return np.concatenate([np.roll(triangles, i, axis=1) for i in range(3)])

def build_lattice(side_n, more_interrupted=False, include_degenerate=False):
  if more_interrupted:
    return build_lattice_more_interrupted(side_n, include_degenerate)
  else:
    return build_lattice_less_interrupted(side_n, include_degenerate)

# PLEASE DON"T READ THIS FUNCTION
def build_lattice_more_interrupted(side_n, include_degenerate=False):
  # LET THERE BE SPAGHETTI
  north_leaf_end_thetas = np.array([140, 360]) * TAU / 360
  south_leaf_end_thetas = np.array([80, 260, 360]) * TAU / 360
  far_south_leaf_end_thetas = np.array([80, 160, 260, 360]) * TAU / 360
  def leaf_stuff(end_thetas):
    widths = np.diff(end_thetas)
    props = widths / end_thetas[-1]
    full_side_n_csum = np.round((end_thetas * side_n) / TAU).astype(np.int64)
    side_ns = np.diff(full_side_n_csum, prepend=0)
    start_thetas = np.zeros_like(end_thetas)
    start_thetas[1:] = end_thetas[:-1]
    #thetass = [np.linspace(start_theta, end_theta, leaf_n, endpoint=False) for start_theta, end_theta, leaf_n in zip(start_thetas, end_thetas, side_ns)]
    mid_thetas = (start_thetas + end_thetas) / 2
    n = len(end_thetas)
    return side_ns, start_thetas, mid_thetas, end_thetas, n
  north_leaf_n, north_start_thetas, north_mid_thetas, north_end_thetas, north_n_leafs = leaf_stuff(north_leaf_end_thetas)
  south_leaf_n, south_start_thetas, south_mid_thetas, south_end_thetas, south_n_leafs = leaf_stuff(south_leaf_end_thetas)
  far_south_leaf_n, far_south_start_thetas, far_south_mid_thetas, far_south_end_thetas, far_south_n_leafs = leaf_stuff(far_south_leaf_end_thetas)

  theta_rows = []
  z_rows = []
  triangle_rows = []
  leaf_specs = [None for _ in range(north_n_leafs)]
  full_theta_row = np.linspace(0, TAU, side_n + 1)
  base_idx = 0
  def add_row(thetas, zs, leaf_idx=None) -> RowSpec:
    nonlocal base_idx
    theta_rows.append(thetas)
    z_rows.append(zs)
    row_n = len(thetas)
    spec = RowSpec(base_idx, row_n)
    base_idx += row_n
    return spec

  if include_degenerate:
    leaf_specs[0] = add_row([0, north_mid_thetas[0]], [1, 1])
    leaf_specs[-1] = add_row([north_mid_thetas[-1], TAU], [1, 1])
  else:
    leaf_specs[0] = add_row([north_mid_thetas[0]], [1])
    leaf_specs[-1] = add_row([north_mid_thetas[-1]], [1])

  for i in range(1, north_n_leafs - 1):
    leaf_specs[i] = add_row([north_mid_thetas[i]], [1])

  # SPAGHAETITI 
  full_theta_row = np.linspace(0, TAU, side_n + 1)

  n_big_rows = int(2 * side_n / TAU) + 1
  dz = 2 / (n_big_rows - 1)
  middle_row = n_big_rows // 2
  for big_row in range(1, n_big_rows - 1):
    z = 1.0 - dz * big_row
    scale = 1 / (1 - z ** 2)
    reps = 2**int(np.log2(scale) / 2 + 0.5)
    for rep in reversed(range(reps)):
      # S P A G H E T T I
      if z < 0:
        rep -= reps - 1
      mini_row_z = z + rep * dz / reps

      toggle_left = z < 0
      #if big_row == middle_row:
      #  zs = np.full([1 + side_n // reps], mini_row_z)
      #  new_leafs = [add_row(full_theta_row, zs)]
      new_leafs = []
      if z > 0.087:
        iter_ = zip(north_leaf_n, north_start_thetas, north_end_thetas)
      elif z >= -0.087:
        iter_ = [(side_n, 0, TAU)]
      elif z >= -0.642:
        iter_ = zip(south_leaf_n, south_start_thetas, south_end_thetas)
      else:
        iter_ = zip(far_south_leaf_n, far_south_start_thetas, far_south_end_thetas)

      for leaf_n, start_theta, end_theta in iter_:
        thetas = np.linspace(start_theta, end_theta, leaf_n // reps + 1)
        zs = np.full(thetas.shape, mini_row_z)
        new_leafs.append(add_row(thetas, zs))

      if len(new_leafs) == len(leaf_specs):
        for new, old in zip(new_leafs, leaf_specs):
          triangle_rows.append(triangles_for_rows(old, new, toggle_left=toggle_left))
      else:
        triangle_rows.append(triangles_for_multi_rows(leaf_specs, new_leafs, toggle_left=toggle_left))
      leaf_specs = new_leafs

  # MOMS SPAGHETTI
  if include_degenerate:
    r_frst = add_row([0, far_south_mid_thetas[0]], [-1, -1])
    r_last = add_row([far_south_mid_thetas[-1], TAU], [-1, -1])
  else:
    r_frst = add_row([far_south_mid_thetas[0]], [-1])
    r_last = add_row([far_south_mid_thetas[-1]], [-1])
  triangle_rows.append(triangles_for_rows(leaf_specs[0], r_frst, toggle_left=True))
  triangle_rows.append(triangles_for_rows(leaf_specs[-1], r_last, toggle_left=True))

  for i in range(1, far_south_n_leafs - 1):
    r = add_row([far_south_mid_thetas[i]], [-1])
    triangle_rows.append(triangles_for_rows(leaf_specs[i], r, toggle_left=True))

  theta = np.concatenate(theta_rows)
  z = np.concatenate(z_rows)
  triangles = np.concatenate(triangle_rows)

  phi = np.arccos(z)
  x = np.sin(phi) * np.cos(theta)
  y = np.sin(phi) * np.sin(theta)

  euc = np.stack([x, y, z], axis=1)
  sph = np.stack([theta, phi], axis=1)

  return Lattice(
    sph,
    euc,
    triangles,
  )

def build_lattice_less_interrupted(side_n, include_degenerate=False):
  full_theta_row = np.linspace(0, TAU, side_n + 1)

  theta_rows = []
  z_rows = []
  triangles = []

  n_big_rows = int(2 * side_n / TAU) + 1
  dz = 2 / (n_big_rows - 1)
  for big_row in range(n_big_rows):
    if big_row == 0 or big_row == n_big_rows - 1:
      z = 1.0 if big_row == 0 else -1.0
      if include_degenerate:
        # hack in degenerate triangles so traditional projections can fill the whole area
        theta_rows.append([0, TAU / 2, TAU])
        z_rows.append([z, z, z])
      else:
        theta_rows.append([TAU / 2])
        z_rows.append([z])
      continue
    z = 1.0 - dz * big_row
    # we want to keep darclen/dtheta to be roughly the same as
    # darclen/dz. The former is sqrt(1 - z^2) and the latter is
    # the reciprocal of that; so we need to split the row up
    # when this "scale" gets too large
    scale = 1 / (1 - z ** 2)
    
    reps = 2**int(np.log2(scale) / 2 + 0.5)
    for rep in reversed(range(reps)):
      if z < 0:
        rep -= reps - 1
      theta_rows.append(full_theta_row[::reps])
      mini_row_z = z + rep * dz / reps
      z_rows.append(np.full([1 + side_n // reps], mini_row_z))

  n_rows = len(theta_rows)
  base_i = 0
  for row_i in range(n_rows - 1):
    row_n_0 = len(theta_rows[row_i])
    row_n_1 = len(theta_rows[row_i + 1])
    triangles.append(triangles_for_rows(
      RowSpec(base_i, row_n_0),
      RowSpec(base_i + row_n_0, row_n_1),
      toggle_left=z_rows[row_i][0] < 0,
    ))
    base_i += row_n_0

  theta = np.concatenate(theta_rows)
  z = np.concatenate(z_rows)
  triangles = np.concatenate(triangles)

  phi = np.arccos(z)
  x = np.sin(phi) * np.cos(theta)
  y = np.sin(phi) * np.sin(theta)

  euc = np.stack([x, y, z], axis=1)
  sph = np.stack([theta, phi], axis=1)

  return Lattice(
    sph,
    euc,
    triangles,
  )
