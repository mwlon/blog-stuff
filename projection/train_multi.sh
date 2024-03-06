#!/bin/bash
set -eoux pipefail

# add more proportions here
for p in 001 999; do
  if [[ $p -lt 150 || $p -gt 850 ]]; then
    iters=8000
  else
    iters=4000
  fi
  if [[ $p -lt 925 ]]; then
    opt=lbfgs
    lr=0.03
  else
    opt=adam
    if [[ $p -lt 970 ]]; then
      lr=0.002
    elif [[ $p -lt 995 ]]; then
      lr=0.001
    else
      lr=0.0002
    fi
  fi
  python train.py \
    --side-n 240 \
    --n-iters $iters \
    --name Martin_$p \
    --area-loss-prop 0.$p \
    --opts $opt \
    --base-lr $lr
done
