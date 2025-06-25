from typing import Any
import os
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
from jax import jit, grad, vmap
from tabulate import tabulate
import re

TAU = 2 * jnp.pi

def normalized(v):
  return v / (np.linalg.norm(v, axis=1))[:, None]

def plane_angle_between(u, v):
  dot = np.sum(normalized(u) * normalized(v), axis=1)
  dot[dot > 1] = 1
  dot[dot < -1] = -1
  return TAU / 2 - np.arccos(dot)

def arclength_between(u, v):
  euc_displacement = u - v
  euc_norm2 = np.sum(euc_displacement * euc_displacement, axis=1)
  return np.arccos(1 - euc_norm2 / 2)

def calc_areas_angles_lengths(euc, triples):
  euc_triples = np.take_along_axis(euc[None, :, :], triples[:, :, None], axis=1)
  plane_uv = np.cross(euc_triples[:, 1], euc_triples[:, 0])
  plane_vw = np.cross(euc_triples[:, 2], euc_triples[:, 1])
  plane_wu = np.cross(euc_triples[:, 0], euc_triples[:, 2])
  angles = plane_angle_between(plane_vw, plane_uv)
  angles_u = plane_angle_between(plane_wu, plane_uv)
  angles_w = plane_angle_between(plane_vw, plane_wu)
  areas = angles + angles_u + angles_w - TAU / 2
  uv_length = arclength_between(euc_triples[:, 0], euc_triples[:, 1])
  wv_length = arclength_between(euc_triples[:, 2], euc_triples[:, 1])
  return areas, angles, uv_length, wv_length

def calc_inv_atlas(angles, uv_length, wv_length):
  # since we don't care about rotation, we choose a basis
  # such that atlas is a batch of matrices of the form
  # | a  b |
  # | 0  c |
  a = uv_length
  b = wv_length * np.cos(angles)
  c = wv_length * np.sin(angles)
  inv_atlas = np.zeros([len(angles), 2, 2])
  inv_atlas[:, 0, 0] = 1 / a
  inv_atlas[:, 0, 1] = -b / (a * c)
  inv_atlas[:, 1, 1] = 1 / c
  return inv_atlas

def ratio_loss(r):
  return r ** 1 + 1 / r ** 1

def distortion_singular_values(distortion):
  a = distortion[:, 0, 0]
  b = distortion[:, 0, 1]
  c = distortion[:, 1, 0]
  d = distortion[:, 1, 1]
  norm2 = jnp.sum(distortion * distortion, axis=(1, 2))
  desc = jnp.sqrt((a * a + b * b - c * c - d * d) ** 2 + 4 * (a * c + b * d) ** 2)
  val0 = jnp.sqrt((norm2 - desc) / 2)
  val1 = jnp.sqrt((norm2 + desc) / 2)
  return val0, val1

def area_angle_multipliers(distortion):
  val0, val1 = distortion_singular_values(distortion)
  return val0 * val1, val1 / val0

def raw_area_angle_loss(distortion):
  area_mult, angle_mult = area_angle_multipliers(distortion)
  return ratio_loss(area_mult), ratio_loss(angle_mult)

def area_angle_loss(distortion, area_weight, angle_weight):
  area_loss, angle_loss = raw_area_angle_loss(distortion)

  area_loss = jnp.sum(area_loss * area_weight) / jnp.sum(area_weight)
  angle_loss = jnp.sum(angle_loss * angle_weight) / jnp.sum(angle_weight)
  return area_loss, angle_loss

def calc_tangent_vecs(xy, triples):
  xy_triples = jnp.take_along_axis(xy[None, :, :], triples[:, :, None], axis=1)
  uv = xy_triples[:, 0] - xy_triples[:, 1]
  wv = xy_triples[:, 2] - xy_triples[:, 1]
  return jnp.stack([uv, wv], axis=2)

def calc_distortion(inv_atlas, tangent_vecs):
  return tangent_vecs @ inv_atlas

def filter_trained(filter_):
  all_names = sorted([name for name in os.listdir('results') if os.path.isfile(f'results/{name}/triangles.pco')])
  if filter_ is None:
    return all_names
  res = []
  substrings = filter_.split(',')
  for name in all_names:
    for substr in substrings:
      if substr in name or re.match(substr, name):
        res.append(name)
        break
  return res

def calc_euc(sph):
  theta = sph[:, 0]
  phi = sph[:, 1]
  sinphi = np.sin(phi)
  x = sinphi * np.cos(theta)
  y = sinphi * np.sin(theta)
  z = np.cos(phi)
  return np.stack([x, y, z], axis=1)

def rotate(xy, rot):
  rot_mat = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
  return (rot_mat @ xy.transpose()).transpose()

def calc_distortion_dets(distortion):
  assert len(distortion.shape) == 3
  return distortion[:, 0, 0] * distortion[:, 1, 1] - distortion[:, 0, 1] * distortion[:, 1, 0]
  
