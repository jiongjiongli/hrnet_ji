cp /home/data/1824/* /home/data/789/
cp /home/data/1825/* /home/data/789/
cp /home/data/1826/* /home/data/789/
cp /home/data/1827/* /home/data/789/
cp /home/data/1857/* /home/data/789/
cp /home/data/1858/* /home/data/789/
cp /home/data/1859/* /home/data/789/
CUDA_VISIBLE_DEVICES=0 python /project/train/src_repo/train.py
python /project/train/src_repo/toonnx.py