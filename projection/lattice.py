from dataclasses import dataclass
import numpy as np

TAU = 2 * np.pi

@dataclass
class Lattice:
  sph: np.array # n x 2
  euc: np.array # n x 3
  triangles: np.array # k x 3, int
  triples: np.array # 3k x 3, int

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

def triples_for_triangles(triangles):
  return np.concatenate([np.roll(triangles, i, axis=1) for i in range(3)])

def build_lattice(side_n, include_degenerate=False):
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
    #print(big_row, z, scale, reps)
    for rep in reversed(range(reps)):
      if z < 0:
        rep -= reps - 1
      theta_rows.append(full_theta_row[::reps])
      mini_row_z = z + rep * dz / reps
      #print('mini row z', mini_row_z)
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
  triples = triples_for_triangles(triangles)

  return Lattice(
    sph,
    euc,
    triangles,
    triples,
  )
