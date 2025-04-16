CUDA_VISIBLE_DEVICES=2 python Bench/eval/eval_caption.py \
    --data_root /mnt/datalake/DS-lake/vankhoa/M3D-CAP/ \
    --cap_data_path /mnt/datalake/DS-lake/vankhoa/M3D-CAP/M3D_Cap/M3D_Cap.json

CUDA_VISIBLE_DEVICES=2 python Bench/eval/eval_itr.py \
    --data_root /mnt/datalake/DS-lake/vankhoa/M3D-CAP/ \
    --cap_data_path /mnt/datalake/DS-lake/vankhoa/M3D-CAP/M3D_Cap/M3D_Cap.json
