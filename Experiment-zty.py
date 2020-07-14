import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=int, default=0,
                    help='index of config file for experiment')
parser.add_argument('--wdir', type=int, default=0,
                    help='index of work dir for experiment')
# Protect the arguments which are not parsed.
args, unparsed = parser.parse_known_args()

# config dir: '/home/tianyao/Documents/DeeCamp/CarDetectionExample-zty-v2/'
cfg = [
    "local_config/local_atss_r50_fpn_ms12.py",
    "local_config/local_gfl_r50_1x.py"
]

# version log
# work dir: '/home/tianyao/Documents/DeeCamp/output/work_dirs/'
work_dir=[
    # atss
    [
        'output/work_dirs/atss_r50_fpn_ms3',            # stem_channels=32  train=val
        'output/work_dirs/atss_r50_fpn_ms3_stem64',     # stem_channels=64  train=val
        'output/work_dirs/atss_r50_fpn_ms3_10w',        # 10w数据集训练 3 epoch结果
        'output/work_dirs/atss_r50_fpn_ms3_v1',         # 小样本train=1000,val=500

    ],

    # gfl
    [
        'output/work_dirs/gfl_r50_1x_v2/',              # 更改学习律为原始版本     train=val
        'output/work_dirs/gfl_r50_1x/',                 # 学习率按照atss进行调整   train=val
        'output/work_dirs/gfl_r50_ms3_v1',                 # 小样本train=1000,val=500
        'output/work_dirs/gfl_r50_ms3_v2'               # backbone= ResNet50v1d->ResNet50
    ]
]
# print('='*10,"[zty] experiment 1 [ ",
#       time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" ]",'='*10)
# os.system("python ./tools/train.py ./%s --work-dir ../%s --gpus 1" %
#           (cfg[args.cfg],work_dir[args.wdir]))

print('='*10,"[zty] experiment 2 [ ",
      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" ]",'='*10)
os.system("python ./tools/train.py ./%s --work-dir ../%s --gpus 1" %
          (cfg[1],work_dir[1][3]))