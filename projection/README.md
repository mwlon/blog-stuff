# Learned Map Projections

[blog post part 1](https://graphallthethings.com/posts/map-projections-1)

## Using this

There are multiple entry points:

* `train.py`: trains a map projection from scratch and saves the results into an image and .pco files
* `train_multi.sh`: trains multiple models with different area loss proportions
* `scatter_loss.py`: computes and plots the areal and angular loss for both traditional projections and loaded map projections
* `interpolate.py`: produces maps by blending a sequences of saved map projections. Their lattices must be identical.
* `encode_video.sh`: produces a web-compatible video from a sequence of interpolated frames
* `viz_lattice.py`: makes 3D plots of the lattice
* `plot_map.py`: plots the map and optionally distortion of any projection

Note: traditional map projections are never serialized, since that introduces quantization loss.
The only reason to discretize them is to plot them.
Instead, their losses are computed using functions from `traditional.py` which take their exact Jacobians.

