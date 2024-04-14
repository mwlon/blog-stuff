from matplotlib import pyplot as plt
import argparse
import numpy as np
#from matplotlib.collections import LineCollection
import lattice

parser = argparse.ArgumentParser()
parser.add_argument(
  '--name',
  type=str,
  help='name to save results under',
)
parser.add_argument(
  '--side-n',
  type=int,
  default=104,
  help='max width of lattice',
)
parser.add_argument(
  '--x-cutoff',
  type=float,
  default=-1.01,
  help='min x coordinate',
)
parser.add_argument(
  '--z-cutoff',
  type=float,
  default=-1.01,
  help='min z coordinate',
)
parser.add_argument(
  '--blue-below',
  type=float,
  default=-1,
  help='make things blue below this z coord',
)
parser.add_argument(
  '--green-below',
  type=float,
  default=-1,
  help='make things green below this z coord',
)
parser.add_argument(
  '--viewpoint',
  type=str,
  help='which direction to view from',
)
parser.add_argument(
  '--more-interrupted',
  action='store_true',
)

args = parser.parse_args()

pts = lattice.build_lattice(args.side_n, more_interrupted=args.more_interrupted)
print('n pts', pts.euc.shape[0])
print('n triangles', pts.triangles.shape[0])
print('n triples', pts.triples.shape[0])
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for triple in pts.triples:
  euc = pts.euc[triple]
  min_z = np.min(euc, axis=0)[2]
  if min_z < args.z_cutoff:
    continue
  if np.min(euc, axis=0)[0] < args.x_cutoff:
    continue
  if min_z < args.blue_below:
    color = 'blue'
  elif min_z < args.green_below:
    color = 'green'
  else:
    color = 'red'
  ax.plot(*np.transpose(euc), color=color)

if args.viewpoint == 'top':
  ax.view_init(elev=90, azim=-90)
elif args.viewpoint == 'front':
  ax.view_init(elev=0, azim=0)
else:
  raise Exception('unknown viewpoint')
ax.set_proj_type('ortho')
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'results/view_{args.name}')
plt.show()
