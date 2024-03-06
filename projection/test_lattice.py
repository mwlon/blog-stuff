from lattice import *
import pytest

def test_triangles_for_rows():
  triangles = triangles_for_rows(RowSpec(7, 3), RowSpec(10, 3), toggle_left=True)
  np.testing.assert_array_equal(triangles, [
    [7, 11, 8],
    [8, 11, 9],
    [10, 7, 11],
    [11, 9, 12],
  ])

  triangles = triangles_for_rows(RowSpec(0, 3), RowSpec(3, 5), toggle_left=True)
  np.testing.assert_array_equal(triangles, [
    [0, 4, 1],
    [1, 6, 2],
    [3, 0, 4],
    [4, 1, 5],
    [5, 1, 6],
    [6, 2, 7],
  ])

  triangles = triangles_for_rows(RowSpec(10, 5), RowSpec(0, 3), toggle_left=True)
  np.testing.assert_array_equal(triangles, [
    [10, 0, 11],
    [11, 1, 12],
    [12, 1, 13],
    [13, 2, 14],
    [0, 11, 1],
    [1, 13, 2],
  ])

  triangles = triangles_for_rows(RowSpec(0, 1), RowSpec(1, 3), toggle_left=True)
  np.testing.assert_array_equal(triangles, [
    [1, 0, 2],
    [2, 0, 3],
  ])

  triangles = triangles_for_rows(RowSpec(0, 3), RowSpec(3, 9), toggle_left=True)
  print(triangles)
  np.testing.assert_array_equal(triangles, [
    [0, 5, 1],
    [1, 9, 2],
    [3, 0, 4],
    [4, 0, 5],
    [5, 1, 6],
    [6, 1, 7],
    [7, 1, 8],
    [8, 1, 9],
    [9, 2, 10],
    [10, 2, 11],
  ])

@pytest.mark.parametrize('side_n', [4, 48])
def test_lattice_properties(side_n):
  lattice = build_lattice(side_n)
  assert len(lattice.sph.shape) == 2
  assert len(lattice.euc.shape) == 2
  assert lattice.sph.shape[0] == lattice.euc.shape[0]
  assert lattice.sph.shape[1] == 2
  assert lattice.euc.shape[1] == 3
  assert len(lattice.triples.shape) == 2
  assert lattice.triples.shape[1] == 3
