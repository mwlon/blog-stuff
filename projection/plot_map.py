import os
import argparse
from map_utils import plot_map, plot_mults
from lattice import build_lattice, triples_for_triangles
import traditional
import serialization
from math_utils import calc_areas_angles_lengths, area_angle_multipliers, calc_inv_atlas, calc_euc, calc_tangent_vecs, calc_distortion
import numpy as np

MAX_MULT = 100.0

parser = argparse.ArgumentParser()
parser.add_argument(
  '--all-traditional',
  action='store_true',
  help='plot all traditional projections',
)
parser.add_argument(
  '--trained',
  type=str,
  help='name of trained map projection to draw',
)
parser.add_argument(
  '--side-n',
  type=int,
  default=256,
  help='max width of lattice to use for traditional projections',
)
parser.add_argument(
  '--show',
  action='store_true',
  help='whether to show or not',
)
parser.add_argument(
  '--distortion',
  action='store_true',
  help='whether to make area and angle distortion plots',
)
parser.add_argument(
  '--draw-lines',
  action='store_true',
  help='whether to overlay triangles on map',
)
parser.add_argument(
  '--scale',
  type=int,
  default=1024,
  help='how large of a map to make',
)

args = parser.parse_args()
show = args.show

def safe_mult(mult, default):
  return np.minimum(np.nan_to_num(mult, nan=default, posinf=default, neginf=default), MAX_MULT)

if args.all_traditional:
  lattice = build_lattice(args.side_n, include_degenerate=True)
  sph = lattice.sph

  for projection in traditional.projections:
    print(f'{projection.name}...')
    os.makedirs(f'results/{projection.name}', exist_ok=True)
    xy, filtered_triangles = traditional.project(projection, sph, lattice.triangles)
    plot_map(projection.name, sph, xy, filtered_triangles, show=show, draw_lines=args.draw_lines, scale=args.scale)

    if args.distortion:
      triples = triples_for_triangles(lattice.triangles)
      inv_metric = traditional.calc_inv_metric(sph)
      distortion = traditional.calc_distortion(projection, sph, inv_metric)[triples[:, 1]]
      area_mults, angle_mults = area_angle_multipliers(distortion)
      area_mults = safe_mult(area_mults, default=1.0 if projection.equal_area else MAX_MULT)
      angle_mults = safe_mult(angle_mults, default=1.0 if projection.conformal else MAX_MULT)
      plot_mults(projection.name, xy, lattice.triangles, area_mults, angle_mults, show=show)

if args.trained is not None:
  name = args.trained
  loaded = serialization.load(name)
  sph = loaded.sph
  xy = loaded.xy
  triangles = loaded.triangles
  plot_map(name, sph, xy, triangles, show=show, draw_lines=args.draw_lines, scale=args.scale)

  if args.distortion:
    euc = calc_euc(sph)
    triples = triples_for_triangles(triangles)
    areas, angles, uv_length, wv_length = calc_areas_angles_lengths(euc, triples)
    inv_atlas = calc_inv_atlas(angles, uv_length, wv_length)
    tangent_vecs = calc_tangent_vecs(xy, triples)
    distortion = calc_distortion(inv_atlas, tangent_vecs)
    area_mults, angle_mults = area_angle_multipliers(distortion)
    plot_mults(name, xy, triangles, area_mults, angle_mults, show=show)
