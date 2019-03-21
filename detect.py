import argparse
import shutil
import time
from pathlib import Path
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *
from collections import OrderedDict

def detect(
        model,
        images = 'input',
        output='output',  # output folder
        img_size=416,
        conf_thres=0.6,
        nms_thres=0.3,
        save_txt=False,
        save_images=True,
        webcam=False
):
    device = torch_utils.select_device()
    if not os.path.exists(output):
        os.makedirs(output)  # make new output folder
    # # Load weights
    # if weights.endswith('.pt'):  # pytorch format
    #     if weights.endswith('yolov3.pt') and not os.path.exists(weights):
    #         if (platform == 'darwin') or (platform == 'linux'):
    #             os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
    #     model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    # else:  # darknet format
    #     load_darknet_weights(model, weights)

    model.to(device).eval()

    # Set Dataloader
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = ['boat','person']
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()
        if webcam:
            print('webcam frame %g: ' % (i + 1), end='')
        else:
            print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

        if len(pred) > 0:
            # Run NMS on predictions
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            # Print results to screen
            unique_classes = detections[:, -1].cpu().unique()
            for c in unique_classes:
                n = (detections[:, -1].cpu() == c).sum()

                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write('%g %g %g %g %g %g\n' %
                                   (x1, y1, x2, y2, cls, cls_conf * conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[int(cls)])

        dt = time.time() - t
        print('Done. (%.3fs)' % dt)

        if save_images:  # Save generated image with detections
            cv2.imwrite(save_path, im0)

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

    if save_images and (platform == 'darwin'):  # linux/macos
        os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/xy_yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='input', help='path to images')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.70, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
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
        detect(
            model
        )
