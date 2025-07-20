side=1728
python train.py \
  --side-n $side \
  --n-iters 30000 \
  --name water_$side \
  --area-loss-prop 0.7 \
  --opt adam \
  --schedule cosine \
  --base-lr 0.001 \
  --water-angle-loss-mult 0.01 \
  "$@"
