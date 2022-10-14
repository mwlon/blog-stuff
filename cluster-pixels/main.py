import numpy as np
from collections import Counter
import sys
import cv2
import matplotlib
matplotlib.use('MacOSX')
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
  'source_filename',
  type=str,
  help='path to the source image',
)
parser.add_argument(
  '--dest_filename',
  dest='dest_filename',
  type=str,
  help='path to the output image',
  default='out.png',
)
parser.add_argument(
  '--k',
  dest='k',
  type=int,
  help='how many clusters to make',
  default=50,
)
parser.add_argument(
  '--max_iters',
  dest='max_iters',
  type=int,
  help='how many iterations to terminate at',
  default=30,
)
parser.add_argument(
  '--pos_weight',
  dest='pos_weight',
  type=float,
  help='how much to weight coordinate position relative to color values',
  default=1.0,
)
parser.add_argument(
  '--agree_ratio',
  dest='agree_ratio',
  type=float,
  help='how aggressively to smooth image after clustering',
  default=0.03,
)
parser.add_argument(
  '--mode',
  type=str,
  help='color space to use; one of "hsv", "lab", "luv", "rgb"',
  default='lab',
)
args = parser.parse_args()

img = cv2.imread(args.source_filename)
assert img is not None
k = args.k
max_iters = args.max_iters
pos_weight = args.pos_weight
assert pos_weight > 0
agree_ratio = args.agree_ratio
assert agree_ratio >= 0
mode = args.mode.lower()

print('raw shape', img.shape)
max_area = 500 * 500.0
scale = np.sqrt(max_area / (img.shape[0] * img.shape[1]))
if scale < 1:
  new_shape = (int(scale * img.shape[1]), int(scale * img.shape[0]))
  img = cv2.resize(img, new_shape)
  print('resized shape', img.shape)

h, w, _ = img.shape
#cv2.imwrite('resized.png', img)

if mode == 'hsv':
  h_scale = 10.0
  s_scale = 2.0
  v_scale = 1.0 # actually value ^ 2 scale
  n_channel = 4
  def process_img(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    interm = np.zeros([h, w, n_channel])
    hue_radians = hsv[:, :, 0] * np.pi / 90.0
    value = (hsv[:, :, 2] / 255.0) ** 2
    v_sat = hsv[:, :, 1] / 255.0 * value
    interm[:, :, 0] = np.cos(hue_radians) * v_sat * h_scale
    interm[:, :, 1] = np.sin(hue_radians) * v_sat * h_scale
    interm[:, :, 2] = v_sat * s_scale
    interm[:, :, 3] = value * v_scale
    return interm
  
  def unprocess_img(interm):
    hsv = np.zeros([interm.shape[0], interm.shape[1], 3], dtype=np.uint8)
    value = np.sqrt(interm[:, :, 3]) / v_scale
    sat = interm[:, :, 2] / s_scale / value
    hsv[:, :, 1] = np.round(sat * 255.0).astype(np.uint8)
    hsv[:, :, 2] = np.round(value * 255.0).astype(np.uint8)
    atan = np.arctan2(interm[:, :, 1], interm[:, :, 0])
    atan2pi = np.where(atan < 0, atan + 2 * np.pi, atan)
    hsv[:, :, 0] = np.round(atan2pi * 90 / np.pi).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
elif mode == 'luv':
  n_channel = 3
  def process_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2Luv) / 255.0
  
  def unprocess_img(interm):
    return cv2.cvtColor((interm * 255.0).astype(np.uint8), cv2.COLOR_Luv2BGR)
elif mode == 'lab':
  n_channel = 3
  def process_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2Lab) / 255.0
  
  def unprocess_img(interm):
    return cv2.cvtColor((interm * 255.0).astype(np.uint8), cv2.COLOR_Lab2BGR)
else:
  r_scale = 1.0
  g_scale = 1.4
  b_scale = 0.6
  n_channel = 3
  def process_img(img):
    interm = img / 255.0
    interm[:, :, 0] *= b_scale
    interm[:, :, 1] *= g_scale
    interm[:, :, 2] *= r_scale
    return interm
  
  def unprocess_img(interm):
    img = interm * 255.0
    img[:, :, 0] /= b_scale
    img[:, :, 1] /= g_scale
    img[:, :, 2] /= r_scale
    return img.astype(np.uint8)

n_dim = n_channel + 2

interm = process_img(img)
h_ratio = np.sqrt(h) / np.sqrt(w)
w_ratio = 1 / h_ratio

# make row data
coords0 = np.zeros([h, 2])
coords0[:, 0] = np.linspace(0.0, h_ratio * pos_weight, h)
coords1 = np.zeros([w, 2])
coords1[:, 1] = np.linspace(0.0, w_ratio * pos_weight, w)
coords = coords0[:, None] + coords1[None, :]

color_rows = np.zeros([h * w, n_dim])
color_rows[:, :n_channel] = np.reshape(interm, [-1, n_channel])
color_rows[:, n_channel:] = np.reshape(coords, [-1, 2])

# k means
n_rows = color_rows.shape[0]
centroid_inds = np.random.choice(np.arange(n_rows), k, replace=False)
centroids = color_rows[centroid_inds]
loss = None
idxs = None
counts = None
epsilon = 1E-8

for i in range(max_iters):
  dists = np.linalg.norm(color_rows[None, :, :] - centroids[:, None, :], axis=2)
  idxs = np.argmin(dists, axis=0)
  min_dists = dists[idxs, range(n_rows)]
  new_loss = np.sum(min_dists * min_dists)
  print(f'loss at {i}: {new_loss}')
  centroids = np.zeros([k, n_dim])
  counts = np.zeros([k])
  for i in range(n_rows):
    centroids[idxs[i]] += color_rows[i]
    counts[idxs[i]] += 1
  centroids = centroids / (counts[:, None] + epsilon)

  if loss is not None:
    if new_loss > loss:
      raise Exception('noooo')
    elif new_loss == loss:
      print('got same loss, exit')
      break
  loss = new_loss

reshaped_idxs = np.reshape(idxs, [h, w])
mean_colors = centroids[:, :n_channel]
out = np.zeros([h, w, n_channel])
for c in range(n_channel):
  out[:, :, c] = np.take(mean_colors[:, c], reshaped_idxs)

kmeans_img = unprocess_img(out)
#cv2.imwrite('kmeans.png', kmeans_img)

def get_neigh(i, j):
  xoff = [0]
  yoff = [0]
  if i > 0:
    yoff.append(-1)
  if i < h - 1:
    yoff.append(1)
  if j > 0:
    xoff.append(-1)
  if j < w - 1:
    xoff.append(1)
  for xo in xoff:
    for yo in yoff:
      if xo or yo:
        yield (i + yo, j + xo)

n_micro_iter = int(max_area * 1.0)
#n_micro_iter = 0
n_changes = 0
for it in range(n_micro_iter):
  if it % 5000 == 0:
    print('smoothing iter', it, '/', n_micro_iter, 'changed', n_changes)
  start_j = it % w
  start_i = (it // w) % h
  start_pt = (start_i, start_j)
  frontier = [start_pt]
  count = 0
  visited = set(frontier)
  while count < len(frontier):
    i, j = frontier[count]
    agree_counts = Counter()
    idx = reshaped_idxs[i, j]
    for oi, oj in get_neigh(i, j):
      other_idx = reshaped_idxs[oi, oj]
      agree_counts[other_idx] += 1
    best_error = float('infinity')
    best_idx = -1
    color = out[i, j, :n_channel]
    for oidx, ocount in agree_counts.items():
      diff = centroids[oidx, :n_channel] - color
      error = np.sum(diff * diff) - agree_ratio * ocount * ocount
      if error < best_error:
        best_error = error
        best_idx = oidx
    if best_idx != idx:
      n_changes += 1
      reshaped_idxs[i, j] = best_idx
      out[i, j] = centroids[best_idx, :n_channel]
      for neigh in get_neigh(i, j):
        if neigh not in visited:
          frontier.append(neigh)
          visited.add(neigh)
    count += 1


out_img = unprocess_img(out)
cv2.imwrite(args.dest_filename, out_img)

def display_img(to_display):
  cv2.imshow('', to_display)
  while True:
    key = cv2.waitKey(0)
    if key in [27, 113]: # esc or q
      break
  
  cv2.destroyAllWindows()

display_img(out_img)

centroid_colors = unprocess_img(np.array([centroids[:, :n_channel]]))[0]
plt.figure(figsize=(6,6))
for i in range(k):
  cluster_size = np.sum(reshaped_idxs == i)
  cluster_y = -centroids[i, n_channel]
  cluster_x = centroids[i, n_channel + 1]
  color_vec = centroid_colors[i] / 255.0
  # print(f'cluster {i} with RGB {color_vec} has size {cluster_size} at {cluster_x} {cluster_y}')
  color_tup = (color_vec[2], color_vec[1], color_vec[0])
  size = cluster_size / float(h * w) * 5000
  plt.axis('off')

  plt.scatter([cluster_x], [cluster_y], color=color_tup, s=size)
plt.savefig('color_palette.png')
plt.show()

