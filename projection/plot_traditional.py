from typing import Any
import json
import numpy as np
from jax import numpy as jnp
import matplotlib as mpl
import cv2
mpl.use('macosx')
from matplotlib import pyplot as plt
from dataclasses import dataclass, field
from datetime import datetime
import argparse
import jax
import optax
from jax import jit, grad, vmap
from tabulate import tabulate
from map_utils import plot_map, plot_mults
from lattice import grid_pts
from math_utils import arclength_between, plane_angle_between, calc_inv_atlas, calc_angles_lengths_areas, area_angle_loss, area_angle_multipliers, calc_tangent_vecs
import time
from tqdm import tqdm
from jaxopt import LBFGS
from traditional import project, projections, calc_filter, filter_triangles, calc_inv_metric, calc_distortion
from lattice import triples_for_triangles
import os

TAU = 2 * jnp.pi

parser = argparse.ArgumentParser()
parser.add_argument(
  '--side-n',
  type=int,
  default=256,
  help='max width of lattice',
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

print('computing grid...')
plot_points = grid_pts(args.side_n, include_degenerate=True)
plot_sph = plot_points.spherical
plot_triangles = plot_points.triangles

#angles, uv_length, wv_length, areas = calc_angles_lengths_areas(euc, triples)
#inv_atlas = calc_inv_atlas(angles, uv_length, wv_length)
#angle_weight = angles
#area_weight = areas

print('plotting...')
for projection in projections:
  print(f'{projection.name}...')
  os.makedirs(f'results/{projection.name}', exist_ok=True)
  xy, filtered_triangles = project(projection, plot_sph, plot_triangles)
  plot_map(projection.name, plot_sph, xy, filtered_triangles, show=args.show, draw_lines=args.draw_lines)

