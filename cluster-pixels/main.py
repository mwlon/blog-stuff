import numpy as np
from collections import Counter
import sys
import cv2
import matplotlib
matplotlib.use('MacOSX')
from matplotlib import pyplot as plt
import argparse
from PIL import Image, ImageCms
import io

epsilon = 1E-8

parser = argparse.ArgumentParser()
parser.add_argument(
  'source_filename',
  type=str,
  help='path to the source image',
)
parser.add_argument(
  '--dest_filename',
  type=str,
  help='path to the output image',
  default='out.png',
)
parser.add_argument(
  '--k_spatial',
  type=int,
  help='number of clusters for 1st step with both space+color channels; defaults to k; must be at least as great as k',
)
parser.add_argument(
  '--k',
  type=int,
  help='final number of clusters after 2nd step with only color channels',
  default=50,
)
parser.add_argument(
  '--max_iters',
  type=int,
  help='how many iterations to terminate at',
  default=30,
)
parser.add_argument(
  '--pos_weight',
  type=float,
  help='how much to weight coordinate position relative to color values',
  default=1.0,
)
parser.add_argument(
  '--agree_ratio',
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

def group_mean(values, idxs, n):
  dtype=values.dtype
  res = np.zeros([n, values.shape[1]], dtype=dtype)
  counts = np.zeros(n, dtype=dtype)
  np.add.at(res, idxs, values)
  np.add.at(counts, idxs, np.ones(idxs.shape[0], dtype=dtype))
  return res / counts[:, None]

def display_img(to_display):
  cv2.imshow('', to_display)
  while True:
    key = cv2.waitKey(0)
    if key in [27, 113]: # esc or q
      break
  
  cv2.destroyAllWindows()

def imread(f):
  return cv2.imread(f)

def imwrite(f, img):
  cv2.imwrite(f, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

img = imread(args.source_filename)
#display_img(img)
assert img is not None
k = args.k
k_spatial = k if args.k_spatial is None else args.k_spatial
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

if mode == 'hsv':
  h_scale = 10.0
  v_scale = 1.0 # actually value ^ 2 scale
  n_channel = 3
  def process_img(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    interm = np.zeros([h, w, n_channel])
    hue_radians = hsv[:, :, 0] * np.pi / 90.0
    value2 = (hsv[:, :, 2] / 255.0) ** 2
    v_sat = hsv[:, :, 1] / 255.0 * value2
    interm[:, :, 0] = np.cos(hue_radians) * v_sat * h_scale
    interm[:, :, 1] = np.sin(hue_radians) * v_sat * h_scale
    interm[:, :, 2] = value2 * v_scale
    return interm
  
  def unprocess_img(interm):
    hsv = np.zeros([interm.shape[0], interm.shape[1], 3], dtype=np.uint8)
    value2 = interm[:, :, 2] / v_scale
    value = np.sqrt(value2)
    sat = np.sqrt(interm[:, :, 0] ** 2 + interm[:, :, 1] ** 2) / h_scale / (value2 + epsilon)
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

#recovered = unprocess_img(process_img(img))
#for i in range(h):
#    for j in range(w):
#        orig_pixel = img[i, j].astype(np.float32)
#        recovered_pixel = recovered[i, j].astype(np.float32)
#        if np.linalg.norm(orig_pixel - recovered_pixel) > 5.0:
#            print(i, j, orig_pixel, recovered_pixel)
#            raise Exception('function is not invertible')


#display_img(np.concatenate([img, recovered], axis=0))
n_dim = n_channel + 2

interm = process_img(img).astype(np.float32)
h_ratio = np.sqrt(h) / np.sqrt(w)
w_ratio = 1 / h_ratio

# make row data
coords0 = np.zeros([h, 2], dtype=np.float32)
coords0[:, 0] = np.linspace(0.0, h_ratio * pos_weight, h)
coords1 = np.zeros([w, 2], dtype=np.float32)
coords1[:, 1] = np.linspace(0.0, w_ratio * pos_weight, w)
coords = coords0[:, None] + coords1[None, :]

x_spatial = np.empty([h * w, n_dim], dtype=np.float32)
x_spatial[:, :n_channel] = np.reshape(interm, [-1, n_channel])
x_spatial[:, n_channel:] = np.reshape(coords, [-1, 2])

def k_means(x, k, max_iters):
  n_rows = x.shape[0]
  centroid_inds = np.random.choice(np.arange(n_rows), k, replace=False)
  centroids = x[centroid_inds]
  loss = None
  idxs = None
  x_norm2 = np.linalg.norm(x, axis=1) ** 2
  
  for i in range(max_iters):
    centroid_norm2 = np.linalg.norm(centroids, axis=1) ** 2
    dists2 = x_norm2[:, None] + centroid_norm2[None, :] - 2 * x @ centroids.T
    idxs = np.argmin(dists2, axis=1)
    min_dists2 = dists2[range(n_rows), idxs]
    new_loss = np.sum(min_dists2)
    print(f'loss at {i}: {new_loss}')
    centroids = group_mean(x, idxs, k)
  
    if loss is not None:
      if new_loss > loss * 1.01:
        raise Exception(f'noooo, {loss} became {new_loss}')
      elif new_loss >= loss:
        print('got same loss, exit')
        break
    loss = new_loss
  return idxs, centroids

def display_centroids(centroids, reshaped_idxs, name):
  k = centroids.shape[0]
  centroid_colors = unprocess_img(np.array([centroids]))[0]
  plt.figure(figsize=(6,6))
  plt.xlim(0, np.max(x_spatial[:, n_channel + 1]))
  plt.ylim(-np.max(x_spatial[:, n_channel]), 0)

  centroid_xy = np.zeros([k, 2])
  centroid_count = np.zeros(k)
  np.add.at(centroid_xy, idxs, x_spatial[:, n_channel:])
  np.add.at(centroid_count, idxs, np.ones(h * w))
  centroid_xy /= centroid_count[:, None]
  for i in range(k):
    cluster_size = centroid_count[i]
    cluster_y = -centroid_xy[i, 0]
    cluster_x = centroid_xy[i, 1]
    color_vec = centroid_colors[i] / 255.0
    # print(f'cluster {i} with RGB {color_vec} has size {cluster_size} at {cluster_x} {cluster_y}')
    color_tup = (color_vec[2], color_vec[1], color_vec[0])
    size = cluster_size / float(h * w) * 5000
    plt.axis('off')
  
    plt.scatter([cluster_x], [cluster_y], color=color_tup, s=size, alpha=0.3)
  plt.savefig(f'{name}.png')
  plt.show()

idxs, centroids = k_means(x_spatial, k_spatial, args.max_iters)

# remove spatial dimensions
centroids = centroids[:, :n_channel]
if k < k_spatial:
  display_centroids(
    centroids,
    np.reshape(idxs, [h, w]),
    'spatial_centroids',
  )
  print('FURTHER CLUSTERING COLOR CENTROIDS')
  filter_idxs, filter_centroids = k_means(
    centroids,
    k,
    100,
  )
  idxs = np.take(filter_idxs, idxs)
  centroids = group_mean(x_spatial[:, :n_channel], idxs, k)

reshaped_idxs = np.reshape(idxs, [h, w])
out = np.zeros([h, w, n_channel])
for c in range(n_channel):
  out[:, :, c] = np.take(centroids[:, c], reshaped_idxs)

kmeans_img = unprocess_img(out)
imwrite('kmeans.png', kmeans_img)

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
      diff = centroids[oidx] - color
      error = np.sum(diff * diff) - agree_ratio * ocount * ocount
      if error < best_error:
        best_error = error
        best_idx = oidx
    if best_idx != idx:
      n_changes += 1
      reshaped_idxs[i, j] = best_idx
      out[i, j] = centroids[best_idx]
      for neigh in get_neigh(i, j):
        if neigh not in visited:
          frontier.append(neigh)
          visited.add(neigh)
    count += 1


out_img = unprocess_img(out)
imwrite(args.dest_filename, out_img)

display_img(out_img)
display_centroids(centroids, reshaped_idxs, 'final_centroids')

