set -aux

TXT_ROOT="results/default/"
SAVE_ROOT="results/picture/"

python3 check/draw.py \
    --txt_root $TXT_ROOT \
    --save_root $SAVE_ROOT \