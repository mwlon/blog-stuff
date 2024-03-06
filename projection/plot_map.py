import os
import argparse
from map_utils import plot_map, plot_mults
from lattice import build_lattice
import math_utils
import traditional
import serialization

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
  '--distortion-plots',
  action='store_true',
  help='whether to make area and angle distortion plots',
)
parser.add_argument(
  '--draw-lines',
  action='store_true',
  help='whether to overlay triangles on map',
)

args = parser.parse_args()


if args.all_traditional:
  lattice = build_lattice(args.side_n, include_degenerate=True)

  for projection in traditional.projections:
    print(f'{projection.name}...')
    os.makedirs(f'results/{projection.name}', exist_ok=True)
    xy, filtered_triangles = traditional.project(projection, lattice.sph, lattice.triangles)
    plot_map(projection.name, lattice.sph, xy, filtered_triangles, show=args.show, draw_lines=args.draw_lines)

if args.trained is not None:
  name = args.trained
  loaded = serialization.load(name)
  plot_map(name, loaded.sph, loaded.xy, loaded.triangles, show=args.show, draw_lines=args.draw_lines)
