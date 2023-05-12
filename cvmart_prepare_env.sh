
# Code
rm -rf /project/train/src_repo/hrnet_ji

cd /project/train/src_repo
!git clone https://gitee.com/jiongjiongli/hrnet_ji.git

# Pretrain  model weight
cd /project/train/src_repo/hrnet_ji/
cp /project/train/src_repo/hrnet_ji/start_train.sh /project/train/src_repo/start_train.sh

cd /project/train/src_repo/hrnet_ji
mkdir model_data
!wget -P /project/train/src_repo/hrnet_ji/model_data/ https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-36511-files/198f9610-4d77-473c-8565-7d52eb3ca22f/hrnetv2_w32_weights_voc.pth

# Train
pip install --upgrade pip -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
!pip uninstall -y setuptools
!pip install setuptools==59.5.0 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
cd /project/train
!bash /project/train/src_repo/start_train.sh
rm -rf /project/train/models/*
