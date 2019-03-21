import argparse
import json
import time
from pathlib import Path

from models import *
from utils.datasets import *
from utils.utils import *
from collections import OrderedDict

def test(
        model,
        data_cfg,
        batch_size=24,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.2,
        nms_thres=0.45,
        save_json=False,
        model=None
):
    device = torch_utils.select_device(0)
    model.to(device).eval()
    # Configure run
    data_cfg_dict = parse_data_cfg(data_cfg)
    nC = int(data_cfg_dict['classes'])  # number of classes (80 for COCO)


    # Get dataloader
    # dataloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path), batch_size=batch_size)
    dataloader = LoadImagesAndLabels(data_cfg_dict['folder'],data_cfg_dict['valid'],batch_size, img_size)

    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    mP, mR, mAPs, TP, jdict = [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    coco91class = coco80_to_coco91_class()
    for (imgs, targets, paths, shapes) in dataloader:
        targets = targets.to(device)
        t = time.time()
        output = model(imgs.to(device))
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)

        # Compute average precision for each sample
        for si, detections in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            seen += 1

            if detections is None:
                # If there are labels but no detections mark as zero AP
                if len(labels) != 0:
                    mP.append(0), mR.append(0), mAPs.append(0)
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections[(-detections[:, 4]).argsort()]

            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                box = detections[:, :4].clone()  # xyxy
                scale_coords(img_size, box, shapes[si])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner

                # add to json dictionary
                for di, d in enumerate(detections):
                    jdict.append({
                        'image_id': int(Path(paths[si]).stem.split('_')[-1]),
                        'category_id': coco91class[int(d[6])],
                        'bbox': [float3(x) for x in box[di]],
                        'score': float3(d[4] * d[5])
                    })

            # If no labels add number of detections as incorrect
            correct = []
            if len(labels) == 0:
                # correct.extend([0 for _ in range(len(detections))])
                mP.append(0), mR.append(0), mAPs.append(0)
                continue
            else:
                # Extract target boxes as (x1, y1, x2, y2)
                target_box = xywh2xyxy(labels[:, 1:5]) * img_size
                target_cls = labels[:, 0]

                detected = []
                for *pred_box, conf, cls_conf, cls_pred in detections:
                    # Best iou, index between pred and targets
                    iou, bi = bbox_iou(pred_box, target_box).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and cls_pred == target_cls[bi] and bi not in detected:
                        correct.append(1)
                        detected.append(bi)
                    else:
                        correct.append(0)

            # Compute Average Precision (AP) per class
            AP, AP_class, R, P = ap_per_class(tp=np.array(correct),
                                              conf=detections[:, 4].cpu().numpy(),
                                              pred_cls=detections[:, 6].cpu().numpy(),
                                              target_cls=target_cls.cpu().numpy())

            # Accumulate AP per class
            AP_accum_count += np.bincount(AP_class, minlength=nC)
            AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

            # Compute mean AP across all classes in this image, and append to image list
            mP.append(P.mean())
            mR.append(R.mean())
            mAPs.append(AP.mean())

            # Means of all images
            mean_P = np.mean(mP)
            mean_R = np.mean(mR)
            mean_mAP = np.mean(mAPs)

        # Print image mAP and running mean mAP
        print(('%11s%11s' + '%11.3g' * 4 + 's') %
              (seen, dataloader.nF, mean_P, mean_R, mean_mAP, time.time() - t))

    # Print mAP per class
    print('\nmAP Per Class:')
    for i, c in enumerate(load_classes(data_cfg_dict['names'])):
        print('%15s: %-.4f' % (c, AP_accum[i] / (AP_accum_count[i] + 1E-16)))
    # Return mAP
    return mean_P, mean_R, mean_mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int,default=24, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/xy_yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/xy.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    opt = parser.parse_args()
    print(opt, end='\n\n')
    # Initialize model
    model = Darknet(opt.cfg, opt.img_size)
    # Load weights
    weights = 'weights/best.pt'
    if weights.endswith('.pt'):  # pytorch format
        state_dict = torch.load(weights, map_location='cpu')['model']
       
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' in k:
                namekey = k[7:] # remove `module.`
                new_state_dict[namekey] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:  # darknet format
        load_darknet_weights(model, weights)
    with torch.no_grad():
        mAP = test(
            model,
            opt.data_cfg,
        )

#       Image      Total          P          R        mAP  # YOLOv3 320
#          32       5000       0.66      0.597      0.591
#          64       5000      0.664       0.62      0.604
#          96       5000      0.653      0.627      0.614
#         128       5000      0.639      0.623      0.607
#         160       5000      0.642       0.63      0.616
#         192       5000      0.651      0.636      0.621

#       Image      Total          P          R        mAP  # YOLOv3 416
#          32       5000      0.635      0.581       0.57
#          64       5000       0.63      0.591      0.578
#          96       5000      0.661      0.632      0.622
#         128       5000      0.659      0.632      0.623
#         160       5000      0.665       0.64      0.633
#         192       5000       0.66      0.637       0.63

#       Image      Total          P          R        mAP  # YOLOv3 608
#          32       5000      0.653      0.606      0.591
#          64       5000      0.653      0.635      0.625
#          96       5000      0.655      0.642      0.633
#         128       5000      0.667      0.651      0.642
#         160       5000      0.663      0.645      0.637
#         192       5000      0.663      0.643      0.634
