from pcodec import ChunkConfig, standalone
import os
import numpy as np
from dataclasses import dataclass
import numpy as np
from lattice import triples_for_triangles, Lattice

def save(name, sph, triangles, xy):
  d = f'results/{name}'
  os.makedirs(d, exist_ok=True)
  for series, vals in [
    ('theta', sph[:, 0]),
    ('phi', sph[:, 1]),
    ('triangles', triangles.flatten()),
    ('x', xy[:, 0]),
    ('y', xy[:, 1]),
  ]:
    vals = np.ascontiguousarray(vals)
    with open(f'{d}/{series}.pco', 'wb') as f:
      f.write(standalone.simple_compress(vals, ChunkConfig()))

@dataclass
class LoadedData:
  sph: np.ndarray
  triangles: np.ndarray
  xy: np.ndarray

  def lattice(self) -> Lattice:
    sph = self.sph
    phi = sph[:, 0]
    theta = sph[:, 1]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    euc = np.stack([x, y, z], axis=1)
    triples = triples_for_triangles(self.triangles)
    return Lattice(sph=sph, euc=euc, triangles=self.triangles, triples=triples)
  
def load(name):
  d = f'results/{name}'
  def decompress(series):
    return standalone.simple_decompress(open(f'{d}/{series}.pco', 'rb').read())
  sph = np.stack([
    decompress('theta'),
    decompress('phi'),
  ], axis=1)
  triangles = decompress('triangles').reshape([-1, 3])
  xy = np.stack([
    decompress('x'),
    decompress('y'),
  ], axis=1)
  return LoadedData(sph, triangles, xy)
