from math_utils import *
import numpy as np
import pytest

TAU = 2 * np.pi

#def test_angles_near_pole():
#  phi = 0.1
#  theta = 0.2
#  euc = np.array([
#    [0, 0, 1],
#    [np.sin(phi), 0, np.cos(phi)],
#    [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
#  ])
#  triples = np.array([
#    [0, 1, 2],
#    [1, 2, 0],
#    [2, 0, 1],
#  ])
#  angles, _, _, _ = calc_angles_lengths_areas(euc, triples)
#  np.testing.assert_allclose(angles, [
#    TAU / 4,
#    TAU / 4,
#    theta,
#  ])

def test_calc_angles_lengths_areas():
  euc = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, np.sqrt(3) / 2, 0.5],
    [-1, 0, 0],
  ])
  triples = np.array([
    [0, 1, 2],
    [1, 0, 2],
    [3, 1, 0],
  ])
  areas, angles, uv_length, wv_length = calc_areas_angles_lengths(euc, triples)
  np.testing.assert_allclose(angles, [
    TAU / 4,
    TAU / 12,
    TAU / 2,
  ])
  np.testing.assert_allclose(uv_length, [
    TAU / 4,
    TAU / 4,
    TAU / 4,
  ])
  np.testing.assert_allclose(wv_length, [
    TAU / 12,
    TAU / 4,
    TAU / 4,
  ])
  np.testing.assert_allclose(areas, [
    TAU / 12,
    TAU / 12,
    np.nan,
  ])

def test_calc_inv_atlas_value():
  angles = [TAU / 8, TAU / 4]
  inv_atlas = calc_inv_atlas(angles, uv_length=1, wv_length=1)
  np.testing.assert_allclose(
    inv_atlas,
    [
      [[1, -1], [0, np.sqrt(2)]],
      [[1, 0], [0, 1]],
    ],
    atol=1E-6
  )

def test_calc_inv_atlas_property():
  np.random.seed(0)
  n = 50
  angles = np.random.uniform(0, TAU, size=n)
  uv_length = np.random.uniform(0, 1, size=n)
  wv_length = np.random.uniform(0, 1, size=n)
  atlas = np.zeros([n, 2, 2])
  atlas[:, 0, 0] = uv_length
  atlas[:, 0, 1] = wv_length * np.cos(angles)
  atlas[:, 1, 1] = wv_length * np.sin(angles)
  inv_atlas = calc_inv_atlas(angles, uv_length, wv_length)
  np.testing.assert_allclose(
    inv_atlas @ atlas,
    np.repeat(np.eye(2)[None, :, :], n, axis=0),
    atol=1E-6
  )

def test_arclength_between():
  np.testing.assert_allclose(
    arclength_between(
      np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
      np.array([[1, 0, 0], [np.sqrt(3) / 2, 0, 0.5], [0, 1, 0]]),
    ),
    [0, TAU / 12, TAU / 4]
  )

def test_singular_values():
  atlas = np.array([
    [[1, 0], [0, 1]],
    [[0, 1], [1, 0]],
    [[1, 0], [0, 0.5]],
    [[1, np.sqrt(3)], [-np.sqrt(3), 1]],
    [[1, 0], [0, 1]],
  ])
  inv_atlas = np.array([np.linalg.inv(x) for x in atlas])
  tangent_vecs = np.array([
    [[1, 0], [0, -3]],
    [[0.5, 0], [0, 0.5]],
    [[1, 0], [0, 1]],
    [[2, np.sqrt(3) * 3], [-2 * np.sqrt(3), 3]],
    [[1, 2], [3, 4]],
  ])
  distortion = calc_distortion(inv_atlas, tangent_vecs)
  val0, val1 = distortion_singular_values(distortion)
  np.testing.assert_allclose(val0, [1, 0.5, 1, 2, np.sqrt(15 - np.sqrt(221))], rtol=1E-6)
  np.testing.assert_allclose(val1, [3, 0.5, 2, 3, np.sqrt(15 + np.sqrt(221))], rtol=1E-6)

def test_ratio_loss():
  assert ratio_loss(0.5) == 2.5
  assert ratio_loss(1) == 2
  assert ratio_loss(2) == 2.5
  np.testing.assert_allclose(ratio_loss(3), 3 + 1.0 / 3)

@pytest.mark.parametrize('theta', [TAU / 12, TAU / 8, TAU / 4, 3 * TAU / 8])
def test_angle_minimizer(theta):
  inv_atlas = calc_inv_atlas(
    np.array([theta, theta, theta]),
    1,
    1,
  )
  tangent_vecs = np.array([
    [[1, np.cos(theta - 0.01)], [0, np.sin(theta - 0.01)]],
    [[1, np.cos(theta)], [0, np.sin(theta)]],
    [[1, np.cos(theta + 0.01)], [0, np.sin(theta + 0.01)]],
  ])
  distortion = calc_distortion(inv_atlas, tangent_vecs)
  _, angle_loss = raw_area_angle_loss(distortion)
  assert angle_loss[0] > angle_loss[1] < angle_loss[2]

def test_calc_tangent_vecs():
  xy = np.array([
    [-0.01, 0.9],
    [0.01, 0.9],
    [0.0, 1.0],
  ])
  triples = np.array([
    [2, 1, 0],
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1],
  ])
  np.testing.assert_allclose(
    calc_tangent_vecs(xy, triples),
    [
      [[-0.01, -0.02], [0.1, 0.0]],
      [[-0.02, -0.01], [0.0, 0.1]],
      [[0.01, -0.01], [-0.1, -0.1]],
      [[0.01, 0.02], [0.1, 0.0]],
    ],
    atol=1E-6
  )

def angle_between(u, v):
  u = u / np.linalg.norm(u)
  v = v / np.linalg.norm(v)
  return np.arccos(np.dot(u, v))

def test_mults_for_triangle():
  euc = np.array([
    [-3.08220700e-01, 3.08220700e-01, -9.00000000e-01],
    [-1.22464680e-16, 1.49975978e-32, -1.00000000e+00],
    [-4.02709752e-01, 1.66807841e-01, -9.00000000e-01],
  ])
  xy = np.array([
    [-3.5688777, -5.5409036],
    [-3.4238954, -5.952549 ],
    [-3.374684 , -5.539749 ],
  ])
  triples = np.array([[0, 1, 2]])#, [1, 2, 0], [2, 0, 1]])

  areas, angles, uv_length, wv_length = calc_areas_angles_lengths(euc, triples)
  inv_atlas = calc_inv_atlas(angles, uv_length, wv_length)
  tangent_vecs = calc_tangent_vecs(xy, triples)
  atlas = np.linalg.inv(inv_atlas[0])
  distortion = np.linalg.solve(atlas.T, tangent_vecs.T).T
  area_mults, angle_mults = area_angle_multipliers(distortion)
