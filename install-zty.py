"""
@author Tianyao Zhang
@desc   该文件用于在modelArts的训练作业中安装mmdetection。
        本地已经配置好环境，而云端每次启动任务均需配置环境。
        pycocotools如果用os.system的指令安装，会出现当前文件无法import的问题，实际是已经安装好的。
        依赖包的wheel或源码已经存放在requirements中，os.system("pip install -r initial-zty.txt")进行安装。
        本地 Deecamp/output 用来临时存储输出结果，并且每次结果覆盖保存
        云端存储在 train_url
        云端无法通过软链接的形式调用其他桶的data  2020-06-26
        修改了config中的samples_per_gpu=8,并创建了本地config文件，以解决显存不足的问题  2020-07-01
        2020-07-06
        更新了mmdetection环境至2.2.0，修改local_config/atss_r50_fpn_ms12.py中的stem_channels=32 for mmdet 2.2.0
        更新了requriements中mmcv=0.6.2  pycocotools=12.0

@date   2020/06/25
说明：
parameters:
    --data_url:     数据输入入口
    --train_url:    存储输出位置
    --cloud:        判断是不是在本地（False表本地运行)

function：

Data：
"""

import os
import time
# import mmcv
import argparse


print('='*10,"[zty] start install-zty [ "+
      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" ]",'='*10)
parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str,
                    default='/home/tianyao/Documents/DeeCamp/data',
                    help='s3 path of dataset')
parser.add_argument('--train_url', type=str, default='../output',
                    help='s3 path of dataset')
parser.add_argument('--cloud', type=bool, default=False,
                    help='not running locally')
# Protect the arguments which are not parsed.
args, unparsed = parser.parse_known_args()

cache_train_url = '/home/tianyao/Documents/DeeCamp/output/'
if(args.cloud):
    # 此处开始，工作目录进入code
    cache_train_url = './cache/'
    os.chdir("code")
    print('-'*20+"show the pip list"+'-'*20)
    os.system("pip list")
    print('-'*20+"show the file list"+'-'*20)
    os.system("ls")
    print('-'*20+"show the pwd"+'-'*20)
    os.system("pwd")
    print('-'*20+"show the nvidia-smi"+'-'*20)
    os.system("nvidia-smi")
    print('='*10,"[zty] start initializing [ "+
          time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" ] ",'='*10)

    # 用来从cocoapi文件夹中安装包。此方法已被requirements中利用源码安装代替
    # os.chdir("./code/cocoapi")
    # os.system("python installcocoapi-zty.py")
    # os.chdir("../../")  # 回到根目录

    os.chdir("./requirements/")
    os.system("pip install -r initial-zty.txt")
    os.chdir("../")         # 回到代码所在根目录 本地：CarDetectionExample-zty；云端：code
    print("[zty] [ "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" ]")

    # os.system("pip install -r requirements/build.txt")
    os.system("pip install -v -e .")

    # 用来debug 查看包是否安装正确
    # print('-'*20+"show the pip list")
    # os.system("pip list")
    # print("successfully installed")
    # print("[ "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" ]")
    # import mmdet
    # print(mmdet.__version__)
    if not os.path.exists("data"):
        print("need training data")
        os.system("ls")
        print('='*10,'loading data start', '='*10,
              time.strftime("[ %Y-%m-%d %H:%M:%S ]", time.localtime()))
        # data_path = './cache/data/'
        data_path = './data/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        import moxing as mox
        mox.file.copy_parallel("s3://detteam/annotations.zip",data_path+"annotations.zip")
        mox.file.copy_parallel("s3://detteam/train.zip", data_path+"train.zip")
        mox.file.copy_parallel("s3://detteam/valid.zip", data_path+"valid.zip")
        mox.file.copy_parallel("s3://detteam/testA.zip", data_path+"testA.zip")
        print('loading data end',os.listdir(data_path))
        os.chdir(data_path)
        os.system('unzip -q valid.zip')
        os.system('unzip -q testA.zip')
        os.system('unzip -q annotations.zip')
        os.system('unzip -q train.zip')
        os.system('rm *.zip')
        os.chdir('../')
        os.system("ls")
        print('='*10,'unzip data end', '='*10,
              time.strftime("[ %Y-%m-%d %H:%M:%S ]", time.localtime()))

if not os.path.exists(cache_train_url):
    os.makedirs(cache_train_url)
print('='*10,"[zty] successfully initialized [ "+
      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" ]",'='*10)

import mmcv  # the mmcv is not installed until here in cloud terminal
timer = mmcv.Timer()

# test the mmdet
print('='*10,"[zty] start mmdet testing [ "+
      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+
      " ] --from start (%.2f s)-- "%timer.since_start(),'='*10)
import sys, os
current_dir = os.getcwd()
sys.path.insert(0, current_dir)
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
# import mmcv

di = './test_img/'
fs = os.listdir(di)

config_file = '%s/local_config/atss_r50_fpn_ms12.py' % current_dir
checkpoint_file = '%s/pretrain_model/atss_r50_fpn_ms12.model' % current_dir
model = init_detector(config_file, checkpoint_file, device='cuda:0')
for idx, f in enumerate(fs):
    if idx < 0 or idx > 2:
        continue
    img = di + f
    print("[zty] "+img)
    result = inference_detector(model, img)
    img = model.show_result(img, result, score_thr=0.3, show=False)

    mmcv.imwrite(img,cache_train_url+"/test_img/"+"test-mmdet-output.png")
    print('[zty] Successful saved (%s) to %s!' % (f,args.train_url))

print('='*10,"[zty] successfully tested mmdet [ "+
      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+
      " ] --costing (%.2f s)-- "%timer.since_last_check(),'='*10)

# train test
print('='*10,"[zty] start traing [ "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+
      " ] --from start (%.2f s)-- "%timer.since_start(),'='*10)
if(args.cloud):
    os.system("python ./tools/train.py ./local_config/atss_r50_fpn_ms12.py --gpus 1")
    mox.file.copy_parallel('./work_dirs/',args.train_url+"/work_dirs/")
    mox.file.copy_parallel(cache_train_url,args.train_url)
else:
    os.system("python ./tools/train.py ./local_config/local_atss_r50_fpn_ms12.py --gpus 1")
    # use the config file for the local terminal

print('='*10,"[zty] successfully trained [ "+
      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+
      " ] --costing (%.2f s)-- "%timer.since_last_check(),'='*10)
