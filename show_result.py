import os
import glob
import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess


compound_coef = 1
force_input_size = None  # set None to use default size
# img_path = "/tmp2/jojo/eye_data/raw/test/n_15.jpg"
img_path = "/tmp2/jojo/eye_data/raw/test"

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['eye']
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                             # replace this part with your project's anchor config
                             ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                             scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

model.load_state_dict(torch.load('logs/d1_2020_05_29_last.pth'))
model.requires_grad_(False)
model.eval()
model = model.cuda()

img_list = glob.glob("./../eye_data/raw/test/*")
for img_path in img_list:
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    out = invert_affine(framed_metas, out)


    for i in range(len(ori_imgs)):
        if len(out[i]['rois']) == 0:
            continue

        for j in range(len(out[i]['rois'])):
            obj = obj_list[out[i]['class_ids'][j]]
            score = float(out[i]['scores'][j])
            (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
            
            color = (232, 162, 0)

            cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), color, 2)
            cv2.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1)
            
            os.makedirs('result_img/eyes_only', exist_ok=True)
            cv2.imwrite(f'result_img/eyes_only/{os.path.basename(img_path)}', ori_imgs[i])
