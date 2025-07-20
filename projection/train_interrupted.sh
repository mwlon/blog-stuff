side=1728
python train.py \
  --side-n $side \
  --n-iters 100000 \
  --name main_$side \
  --area-loss-prop 0.7 \
  --opt lbfgs \
  --schedule const \
  --base-lr 7.7777 \
  --more-interrupted \
  "$@"
