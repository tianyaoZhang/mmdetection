# test the mmdet
import argparse
import sys, os
current_dir = os.getcwd()
sys.path.insert(0, current_dir)
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

cache_train_url = '/home/tianyao/Documents/DeeCamp/output/'
di = './test_img/'
fs = os.listdir(di)

config_file = '%s/local_config/atss_r50_fpn_ms12.py' % current_dir
checkpoint_file = '%s/pretrain_model/atss_r50_fpn_ms12.model' % current_dir

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default=config_file,
                    help='config file')
parser.add_argument('--ckp', type=str, default=checkpoint_file,
                    help='checkpoint_file')
parser.add_argument('--name', type=str, default="test-img-zty",
                    help='checkpoint_file')
parser.add_argument('--show', type=bool, default=False,
                    help='show the results immediately')
# Protect the arguments which are not parsed.
args, unparsed = parser.parse_known_args()

model = init_detector(args.cfg, args.ckp, device='cuda:0')
for idx, f in enumerate(fs):
    if idx < 0 or idx > 2:
        continue
    img = di + f
    print("[zty] "+img)
    result = inference_detector(model, img)
    if args.show:
        model.show_result(img, result, score_thr=0.3, show=True)
    else:
        img = model.show_result(img, result, score_thr=0.3, show=False)
        mmcv.imwrite(img,cache_train_url+"/test_img_zty/"+"%s-%d.png"%(args.name,idx))
        print('[zty] Successful saved (%s) to %s!' % (f,cache_train_url+"/test_img_zty/"))