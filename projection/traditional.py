from dataclasses import dataclass
from jax import numpy as jnp
from jax import vmap, jacobian
import numpy as np
from typing import Callable
import warnings

TAU = 2 * jnp.pi
EPSILON = 1E-7

def natural_earth(v):
  long = v[0] - TAU / 2
  lat = TAU / 4 - v[1]
  l = 0.8707 - 0.131979 * lat ** 2 - 0.013791 * lat ** 4 + 0.003971 * lat ** 10 - 0.001529 * lat ** 12
  d = 1.007226 + 0.015085 * lat ** 2 - 0.044475 * lat ** 6 + 0.028874 * lat ** 8 - 0.005916 * lat ** 10
  return [long * l, lat * d]

def equal_earth(v):
  long = v[0] - TAU / 2
  phi = v[1]
  a1 = 1.340264
  a2 = -0.081106
  a3 = 0.000893
  a4 = 0.003796
  ang = jnp.arcsin(jnp.sqrt(3) * jnp.cos(phi) / 2)
  x = 2 * jnp.sqrt(3) * long * jnp.cos(ang) / 3 / (9 * a4 * ang ** 4 + 7 * a3 * ang ** 6 + 3 * a2 * ang ** 2 + a1)
  y = a4 * ang ** 9 + a3 * ang ** 7 + a2 * ang **3 + a1 * ang
  return [x, y]

def wagner_vi(v):
  long = v[0] - TAU / 2
  lat = TAU / 4 - v[1]
  return [long * jnp.sqrt(1 - 12 * (lat / TAU) **2), lat]

def kavrayskiy_vii(v):
  wagner_x, y = wagner_vi(v)
  return [wagner_x * jnp.sqrt(3) / 2, y]

def winkel_tripel(v):
  long = v[0] - TAU / 2
  phi = v[1]
  phi1cos = 4 / TAU
  phi1 = jnp.arccos(phi1cos)
  phi_sin = jnp.sin(phi)
  alpha = jnp.arccos(phi_sin * jnp.cos(long / 2) * (1 - EPSILON))
  alpha_sinc = jnp.sin(alpha + EPSILON) / (alpha + EPSILON)
  x = long * phi1cos / 2 + phi_sin * jnp.sin(long / 2) / alpha_sinc
  y = (TAU / 4 - phi + jnp.cos(phi) / alpha_sinc) / 2
  return [x, y]

moll_ang = jnp.linspace(-TAU / 4, TAU / 4, 127)
moll_lat = jnp.arcsin((2 * moll_ang + jnp.sin(2 * moll_ang)) / (TAU / 2))

def mollweide(v):
  long = v[0] - TAU / 2
  lat = TAU / 4 - v[1]
  ang = jnp.interp(lat, moll_lat, moll_ang)
  return [4 * jnp.sqrt(2) * long * jnp.cos(ang) / TAU, jnp.sqrt(2) * jnp.sin(ang)]

def mercator(v):
  return [v[0], jnp.log(jnp.tan(TAU / 4 - v[1] / 2))]

def lambert(v):
  return [v[0], jnp.cos(v[1])]

def equirectangular(v):
  return [v[0], TAU / 4 - v[1]]

eckert_ang = jnp.linspace(-1.57, 1.57, 127)
eckert_qty = eckert_ang + jnp.sin(eckert_ang) * jnp.cos(eckert_ang) + 2 * jnp.sin(eckert_ang)
def eckert_iv(v):
  lat = TAU / 4 - v[1]
  ang = jnp.interp((2 + TAU / 4) * jnp.sin(lat), eckert_qty, eckert_ang)
  return [
    4 / np.sqrt(8 * TAU + TAU ** 2) * (v[0] - TAU / 2) * (1 + jnp.cos(ang)),
    2 * np.sqrt(TAU / (8 + TAU)) * jnp.sin(ang)
  ]


@dataclass
class Projection:
  name: str
  f: Callable
  lat_limit: float = 90

projections = [
  # conformal
  Projection('Mercator', mercator, lat_limit = 85),

  # equal area
  Projection('Eckert IV', eckert_iv),
  Projection('Lambert', lambert),
  Projection('Equal Earth', equal_earth),
  Projection('Mollweide', mollweide),

  # compromise
  Projection('Equirectangular', equirectangular),
  Projection('Natural Earth', natural_earth),
  Projection('Wagner VI', wagner_vi),
  Projection('Kavrayskiy VII', kavrayskiy_vii),
  Projection('Winkel Tripel', winkel_tripel),
]

def calc_filter(sph, lat_limit):
  return jnp.abs(sph[:, 1] - TAU / 4) <= lat_limit * TAU / 360

def filter_triangles(triangles, keep):
  for i in range(3):
    triangles = triangles[keep[triangles[:, i]]]
  return triangles

def calc_xy(projection, sph):
  return vmap(lambda v: jnp.stack(projection.f(v)))(sph)

def project(projection, sph, triangles):
  keep = calc_filter(sph, projection.lat_limit)
  triangles = filter_triangles(triangles, keep)
  xy = calc_xy(projection, sph)
  xy = jnp.where(keep[:, None], xy, 0)
  return (xy, triangles)

def calc_inv_metric(sph):
  n = sph.shape[0]
  inv_metric = np.zeros([n, 2, 2])
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    inv_metric[:, 0, 0] = 1 / np.sin(sph[:, 1])
  inv_metric[:, 1, 1] = 1
  return inv_metric

def calc_distortion(projection, sph, inv_metric):
  jac = vmap(jacobian(lambda v: jnp.stack(projection.f(v))))(sph)
  return jac @ inv_metric
 
