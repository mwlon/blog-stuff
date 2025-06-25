import numpy as np
from jax import numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt
import argparse
import jax
import optax
from jax import jit, grad, vmap
from map_utils import plot_map, plot_mults, calc_water_prop
from lattice import build_lattice, triples_for_triangles
from math_utils import arclength_between, plane_angle_between, calc_inv_atlas, calc_areas_angles_lengths, area_angle_loss, area_angle_multipliers, calc_tangent_vecs, calc_distortion, rotate, calc_distortion_dets
import time
from tqdm import tqdm
import traditional
import warnings
import os
import serialization

TAU = 2 * jnp.pi

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

parser = argparse.ArgumentParser()
parser.add_argument(
  '--n-iters',
  type=int,
  default=1000,
  help='number of gradient descent iterations',
)
parser.add_argument(
  '--side-n',
  type=int,
  default=240,
  help='number of lattice points around equator',
)
parser.add_argument(
  '--meme-key',
  type=int,
  default=None,
  help='key for meme random normal initialization',
)
parser.add_argument(
  '--area-loss-prop',
  type=float,
  default=0.5,
  help='what % of loss should be area (as opposed to angle)',
)
parser.add_argument(
  '--base-lr',
  type=float,
  default=0.03,
  help='base learning rate for Adam or SGD; or max step size for LBFGS',
)
parser.add_argument(
  '--opts',
  type=str,
  default='lbfgs',
  help='optimizers to use',
)
parser.add_argument(
  '--name',
  type=str,
  required=True,
  help='name to save results under',
)
parser.add_argument(
  '--show',
  action='store_true',
  help='whether to show or not',
)
parser.add_argument(
  '--draw-lines',
  action='store_true',
  help='whether to overlay triangles on map',
)
parser.add_argument(
  '--log-period',
  type=int,
  default=100,
  help='log loss every this many iterations',
)
parser.add_argument(
  '--save-on-logs',
  action='store_true',
  help='save every time we log loss instead of only at the end',
)
parser.add_argument(
  '--schedule',
  type=str,
  default='const',
  help='learning rate schedule',
)
parser.add_argument(
  '--initial',
  type=str,
  default='natural earth',
  help='initial conditions for map projections',
)
parser.add_argument(
  '--water-angle-loss-mult',
  type=float,
  default=1.0,
  help='how much to multiply angular loss in water by',
)
parser.add_argument(
  '--more-interrupted',
  action='store_true',
)
parser.add_argument(
  '--no-plot',
  action='store_true',
)

args = parser.parse_args()
name = args.name
n_iters = args.n_iters
area_loss_prop = args.area_loss_prop

print('initializing...')
if args.meme_key is None:
  initial = args.initial.lower()
  try:
    initial_projection = next(proj for proj in traditional.projections if proj.name.lower() == initial)
    print('[traditional projection initialization] computing lattice...')
    lattice = build_lattice(args.side_n, more_interrupted=args.more_interrupted)
    params = traditional.calc_xy(initial_projection, lattice.sph)
  except StopIteration:
    proj = serialization.load(initial)
    print('[pretrained projection initialization] reusing lattice...')
    lattice = proj.lattice()
    params = proj.xy
  def calc_xy(params):
    return params
else:
  print('[meme initialization] computing lattice...')
  lattice = build_lattice(args.side_n, more_interrupted=args.more_interrupted)
  initializer = jax.nn.initializers.he_normal()
  hd = 16
  key = args.meme_key
  params = [
    initializer(jax.random.key(key), (2, hd), jnp.float32),
    jax.random.normal(jax.random.key(key + 1), (hd,), jnp.float32),
    initializer(jax.random.key(key + 2), (hd, 2), jnp.float32),
  ]
  def calc_xy(params):
    return jax.nn.tanh(sph / jnp.array([TAU, TAU / 2]) @ params[0] + params[1]) @ params[2]

sph = lattice.sph
euc = lattice.euc
triples = lattice.triples
triangles = lattice.triangles
n = sph.shape[0]

areas, angles, uv_length, wv_length = calc_areas_angles_lengths(euc, triples)
inv_atlas = calc_inv_atlas(angles, uv_length, wv_length)
water_mult = 1 + (args.water_angle_loss_mult - 1) * calc_water_prop(sph, triples)
area_weight = areas * angles
angle_weight = areas * angles * water_mult

triples = jnp.array(triples)

dumb_xy = calc_xy(params)
orig_tangent_vecs = calc_tangent_vecs(dumb_xy, triples)
orig_distortion = calc_distortion(inv_atlas, orig_tangent_vecs)
orig_distortion_dets = calc_distortion_dets(orig_distortion)
orig_distortion_det_signs = orig_distortion_dets >= 0

def loss(params):
  xy = calc_xy(params)
  tangent_vecs = calc_tangent_vecs(xy, triples)
  distortion = calc_distortion(inv_atlas, tangent_vecs)
  area_loss, angle_loss = area_angle_loss(distortion, area_weight, angle_weight)
  return area_loss_prop * area_loss + (1 - area_loss_prop) * angle_loss

def update_is_unsafe(params_updates_iter):
  params, updates, i = params_updates_iter
  xy = calc_xy(params + updates)
  tangent_vecs = calc_tangent_vecs(xy, triples)
  distortion = calc_distortion(inv_atlas, tangent_vecs)
  distortion_det_signs = calc_distortion_dets(distortion) >= 0
  return jnp.array(i < 10) & jnp.any(distortion_det_signs != orig_distortion_det_signs)

def halve_updates(params_updates_iter):
  params, updates, i = params_updates_iter
  return params, updates / 2.0, i + 1

def safely_apply_updates(params, updates):
  params, updates, i = jax.lax.while_loop(update_is_unsafe, halve_updates, (params, updates, 0))

  result = params + 0.8 * updates
  jax.lax.cond(
    i > 0,
    lambda i: jax.debug.print('WARNING: halved gradient {i} times', i=i),
    lambda i: None,
    i,
  )
  return result

if args.schedule == 'cosine':
  schedule = optax.cosine_decay_schedule(args.base_lr, decay_steps=args.n_iters + 1, alpha = 0.01)
elif args.schedule == 'const':
  schedule = args.base_lr
elif args.schedule == 'ramp':
  schedule = optax.linear_schedule(args.base_lr, args.base_lr * 3, transition_steps=2000)
else:
  raise Exception('unknown learning rate schedule')

print('training...')
opts = args.opts.split(',')
n_opts = len(opts)
log_period = args.log_period
for (opt_i, opt_name) in enumerate(opts):
  if opt_name == 'adam':
    opt = optax.adam(schedule, b2=0.99)
  elif opt_name == 'lbfgs':
    opt = optax.lbfgs(schedule)
  elif opt_name == 'sgd':
    opt = optax.sgd(schedule)
  else:
    raise Exception('unknown optimizer')
   
  @jit
  def update(params, opt_state):
    params_grad = grad(loss)(params)
    updates, opt_state = opt.update(params_grad, opt_state, params=params)
    params = safely_apply_updates(params, updates)
    return params, opt_state

  opt_state = opt.init(params)
  print(f'{opt_name}...')
  
  start = (n_iters * opt_i) // n_opts
  end = (n_iters * (opt_i + 1)) // n_opts

  t = time.time()
  def maybe_log(i):
    if i % log_period == 0:
      dt = time.time() - t
      if args.save_on_logs:
        xy = calc_xy(params)
        plot_map(name, sph, xy, triangles, draw_lines=False, show=False, step=i, title='earth')
      loss_float = loss(params).item()
      if not jnp.isfinite(loss_float):
        raise Exception('loss became nan!')
      tqdm.write(f'{i} {dt:.04f} {loss_float}')

  for i in tqdm(range(start, end)):
    maybe_log(i)
    params, opt_state = update(params, opt_state)

  if end > start:
    maybe_log(end)

xy = calc_xy(params)
# rotate so map is straight up
n_pole = np.mean(xy[sph[:, 1] == 0], axis=0)
xy -= n_pole
s_pole = np.mean(xy[sph[:, 1] == TAU / 2], axis=0)
rot = -(TAU / 4 + np.arctan2(s_pole[1], s_pole[0]))
xy = rotate(xy, rot)

print('saving...')
serialization.save(name, sph, triangles, xy)

print(f'xy stdev: {jnp.std(xy[:, 0])} {jnp.std(xy[:, 1])}')
if not args.no_plot:
  print('plotting...')
  os.makedirs(f'results/{name}', exist_ok=True)
  plot_map(name, sph, xy, triangles, draw_lines=args.draw_lines, show=args.show, title='earth')

tangent_vecs = calc_tangent_vecs(xy, triples)
distortion = calc_distortion(inv_atlas, tangent_vecs)
area_loss, angle_loss = area_angle_loss(distortion, area_weight, angle_weight)
print(f'{area_loss.item()=} {angle_loss.item()=}')

