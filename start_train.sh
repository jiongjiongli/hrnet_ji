cd /project/train/src_repo/hrnet_ji
# Check data.
# python /project/train/src_repo/hrnet_ji/experiment.py
CUDA_VISIBLE_DEVICES=0 python /project/train/src_repo/hrnet_ji/train.py
python /project/train/src_repo/hrnet_ji/toonnx.py
