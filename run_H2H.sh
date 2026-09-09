#!/bin/bash
cd /home/fabio/NNS/test/Maleficnet-2
source /home/fabio/NNS/virtualen/bin/activate
GAMMA=${GAMMA:-1e-5}
GPU=${GPU:-1}
PAYLOADS=${PAYLOADS:-"stuxnet destover asprox bladabindi zeus_bank equationdrug zeus_dec kovter cerber ardamax nsis kelihos"}
mkdir -p throwaway/logs

# See run_all.sh for the math: ~192*bytes carriers vs 134M unique-weight pool.
declare -A BAND=( [equationdrug]=0.6 [zeus_dec]=0.65 [kovter]=0.7 [cerber]=0.95 )
BIG="ardamax nsis kelihos"   # payload > pool -> hessian with reuse (band 0.5 and 1.0)

run_arm () {  # $1=payload  $2=arm label  $3..=maleficnet flags
  local P=$1 ARM=$2; shift 2
  local LOG=throwaway/logs/H2H_vgg16_${P}_${ARM}_g${GAMMA}.log
  if grep -aq "Decoding: 100%" "$LOG" 2>/dev/null; then echo "skip $P/$ARM (complet)"; return 0; fi
  echo "=== $P / $ARM / gamma=$GAMMA  $(date) ==="
  CUDA_VISIBLE_DEVICES=$GPU python maleficnet.py --epochs 0 --model vgg16 \
    --payload ${P}.payload --gamma $GAMMA --dataset imagenet --num_classes 1000 \
    --dim 224 --only_pretrained "$@" 2>&1 | tee "$LOG"
  grep -aq "Decoding: 100%" "$LOG" 2>/dev/null
}

echo "== self-check hessian.py =="
python hessian.py || { echo "!! hessian.py self-check FAILED -- abort" >&2; exit 1; }
echo "== smoke: stuxnet hessian =="
run_arm stuxnet hessian --hessian --band 0.5 || {
  echo "!! smoke stuxnet/hessian crashed -- abort before the full sweep" >&2; exit 1; }

for P in $PAYLOADS; do
  run_arm "$P" random
  if echo " $BIG " | grep -q " $P "; then
    run_arm "$P" hessian     --hessian --band 0.5
    run_arm "$P" hessianfull --hessian --band 1.0
  else
    run_arm "$P" hessian --hessian --band "${BAND[$P]:-0.5}"
  fi
done
echo "=== H2H COMPLET $(date) ==="
for f in throwaway/logs/H2H_*.log; do
  echo "== $f"
  grep -aE "test_acc|Injecting on|Curvature carriers|reusing flat|Signal to Noise|System " "$f" | grep -av "%" | head -8
done | tee throwaway/results_H2H.txt
