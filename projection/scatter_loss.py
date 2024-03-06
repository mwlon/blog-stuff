import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import polars as pl
import argparse
import traditional
from lattice import build_lattice, triples_for_triangles
import utils
import serialization

parser = argparse.ArgumentParser()
parser.add_argument(
  '--side-n',
  type=int,
  default=240,
  help='number of lattice points around equator to use for traditional projections',
)
parser.add_argument(
  '--compare-latitude',
  type=float,
  default=85,
  help='max absolute latitude to include in comparison',
)
parser.add_argument(
  '--filter',
  type=str,
  help='substrings to filter trained projections',
)
parser.add_argument(
  '--connect',
  action='store_true',
  help='connect trained projections with a line instead of labeling them',
)
args = parser.parse_args()

names = []
area_losses = []
angle_losses = []
is_trained = []

def add_traditional():
  lattice = build_lattice(args.side_n)
  sph = lattice.sph
  euc = lattice.euc
  triangles = traditional.filter_triangles(lattice.triangles, traditional.calc_filter(sph, args.compare_latitude))
  triples = triples_for_triangles(triangles)
  inv_metric = traditional.calc_inv_metric(sph)
  areas, angles, _, _ = utils.calc_areas_angles_lengths(euc, triples)
  
  for projection in traditional.projections:
    distortion = traditional.calc_distortion(projection, sph, inv_metric)[triples[:, 1]]
    area_loss, angle_loss = utils.area_angle_loss(distortion, areas, angles)
    area_loss = area_loss.item()
    angle_loss = angle_loss.item()

    names.append(projection.name)
    area_losses.append(area_loss)
    angle_losses.append(angle_loss)
    is_trained.append(False)
    print(f'{projection.name},{area_loss},{angle_loss}')

def add_trained():
  for name in utils.filter_trained(args.filter):
    loaded = serialization.load(name)
    sph = loaded.sph
    euc = utils.calc_euc(sph)
    triangles = traditional.filter_triangles(loaded.triangles, traditional.calc_filter(sph, args.compare_latitude))
    triples = triples_for_triangles(triangles)
    areas, angles, uv_len, wv_len = utils.calc_areas_angles_lengths(euc, triples)
    inv_atlas = utils.calc_inv_atlas(angles, uv_len, wv_len)
    tangent_vecs = utils.calc_tangent_vecs(loaded.xy, triples)
    distortion = utils.calc_distortion(inv_atlas, tangent_vecs)
    area_loss, angle_loss = utils.area_angle_loss(distortion, areas, angles)
    area_loss = area_loss.item()
    angle_loss = angle_loss.item()

    names.append(name)
    area_losses.append(area_loss)
    angle_losses.append(angle_loss)
    is_trained.append(True)
    print(f'{name},{area_loss},{angle_loss}')

add_traditional()
add_trained()

area_losses = np.array(area_losses)
angle_losses = np.array(angle_losses)

font = {'family':'sans-serif', 'size':12}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(figsize=(11,8))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.tick_params(axis='both', which='major', labelsize=15)
#ax.locator_params(nbins=5)

offsets = {
  'Mercator': (-0.1, 0.1),
  'Lambert': (0.15, 0),
  'Mollweide': (0.15, 0),
  'Equal Earth': (0.12, 0.07),
  'Kavrayskiy VII': (0.4, 0),
  'Wagner VI': (0.39, 0.2),
  'Winkel Tripel': (0.1, 0.17),
  'Natural Earth': (0.45, 0.33), 'Equirectangular': (0.3, 0),
  'Martin I': (-0.03, -0.1),
  'Eckert IV': (0.2, 0.0)
}
colors = []
line_x = []
line_y = []
for i, name in enumerate(names):
  if not (args.connect and is_trained[i]):
    if name in offsets:
      dx, dy = offsets[name]
    else:
      dx, dy = 0, 0
    x = area_losses[i] + dx
    y = angle_losses[i] + dy
    plt.text(x, y, name, ha='center', va='bottom')
    if dx**2 + dy**2 > 0.01:
      plt.arrow(x, y, -dx, -dy)

  if is_trained[i]:
    colors.append('#00aa55')
  else:
    colors.append('#ee9900')

if args.connect:
  ax.plot(
    area_losses[is_trained],
    angle_losses[is_trained],
    color='#00aa55',
    label='Martin projections',
  )
ax.axhline(2, lw=2, linestyle='dashed', zorder=-100000, label='conformal', color='#5555ee')
ax.axvline(2, lw=2, linestyle='dashed', zorder=-100000, label='equal area', color='#ff0077')
ax.scatter(area_losses, angle_losses, color=colors)
ax.legend()
ax.set_aspect('equal')
ax.set_xlabel('area loss')
ax.set_ylabel('angle loss')
plt.tight_layout()
plt.savefig(f'results/loss_scatter.svg')
plt.show()
