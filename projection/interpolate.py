import numpy as np
import argparse
from map_utils import plot_map
from tqdm import tqdm
import serialization
import os
import utils

parser = argparse.ArgumentParser()
parser.add_argument(
  '--title',
  type=str,
  required=True,
  help='new name to save under',
)
parser.add_argument(
  '--filter',
  type=str,
  required=True,
  help='name or substring of trained projections to load',
)
parser.add_argument(
  '--steps-per',
  type=float,
  default=12,
  help='how many steps go in between each projection',
)

args = parser.parse_args()
title = args.title
steps_per = args.steps_per
names = utils.filter_trained(args.filter)
print(f'filtered to: {names}')

os.makedirs(f'results/{title}', exist_ok=True)

i = 0
prev = None
current = None
for name in tqdm(names):
  tqdm.write(f'{name}...')
  prev = current
  current = serialization.load(name)

  if prev is None:
    # just do the initial frame
    plot_map(title, current.sph, current.xy, current.triangles, show=False, step=i)
    i += 1
  else:
    # do real interpolation
    np.testing.assert_array_equal(prev.sph, current.sph)
    np.testing.assert_array_equal(prev.triangles, current.triangles)
    for j in range(1, steps_per + 1):
      r = j / steps_per
      mixture = r * current.xy + (1 - r) * prev.xy
      plot_map(title, current.sph, mixture, current.triangles, show=False, step=i)
      i += 1
