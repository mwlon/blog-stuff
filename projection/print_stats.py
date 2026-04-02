import numpy as np
import math_utils
import serialization
from matplotlib import pyplot as plt
import scipy.stats
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
  "--trained",
  type=str,
  required=True,
  help="trained projection to print stats for",
)
args = parser.parse_args()


loaded = serialization.load(args.trained)
sph = loaded.sph
euc = math_utils.calc_euc(sph)
lattice = loaded.lattice()
print(lattice.triangles.shape, "triangles")
triples = lattice.triples()
areas, angles, uv_len, wv_len = math_utils.calc_areas_angles_lengths(euc, triples)
inv_atlas = math_utils.calc_inv_atlas(angles, uv_len, wv_len)
tangent_vecs = math_utils.calc_tangent_vecs(loaded.xy, triples)
distortion = math_utils.calc_distortion(inv_atlas, tangent_vecs)
raw_area_loss, raw_angle_loss = math_utils.raw_area_angle_loss(distortion)


def loss(x):
  return x + 1 / x


def inv_loss(c):
  return (c + np.sqrt(c * c - 4)) / 2


percentiles = [0, 0.5, 0.99, 0.999, 0.9999, 1]
quantiles = [loss(1.0), loss(1.1), loss(1.2), loss(1.5), loss(2.0)]
print(inv_loss(np.quantile(raw_area_loss, percentiles)))
print(inv_loss(np.quantile(raw_angle_loss, percentiles)))
print(scipy.stats.percentileofscore(raw_area_loss, quantiles))
print(scipy.stats.percentileofscore(raw_angle_loss, quantiles))
