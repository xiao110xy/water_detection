import argparse
import time

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.summaries import TensorboardSummary
from collections import OrderedDict

def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=True,
        epochs=100,
        batch_size=16,
        accumulate=1,
        multi_scale=False,
        freeze_backbone=False,
):
    # 训练的调试阶段
    run_dir = 'run'
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    lists = os.listdir(run_dir)
    run_path = run_dir+'/'+str(len(lists)+1)
    summary = TensorboardSummary(run_path)
   # writer = summary.create_summary()
    #
    weights = 'weights' + os.sep
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device()

    if multi_scale:
        img_size = 608  # initiate with maximum multi_scale size
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    options = parse_data_cfg(data_cfg)
    

    # Initialize model
    model = Darknet(cfg, img_size)

    # Get dataloader
    dataloader = LoadImagesAndLabels(options['folder'],options['train'],
                 batch_size, img_size, multi_scale=multi_scale, augment=True)

    lr0 = 0.001  # initial learning rate
    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    if False:
        checkpoint = torch.load(latest, map_location='cpu')
        state_dict = checkpoint['model']
       
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' in k:
                namekey = k[7:] # remove `module.`
                new_state_dict[namekey] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        # Load weights to resume from

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device).train()

        # Transfer learning (train only YOLO layers)
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     p.requires_grad = True if (p.shape[0] == 255) else False

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved

    else:
        # Initialize model with backbone (optional)
        # if cfg.endswith('yolov3.cfg'):
        #     load_darknet_weights(model, weights + 'darknet53.conv.74')
        #     cutoff = 75
        # elif cfg.endswith('yolov3-tiny.cfg'):
        #     load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
        #     cutoff = 15

        # Transfer learning (train only YOLO layers)
        checkpoint = torch.load('weights/yolov3.pt', map_location='cpu')
        state_dict = checkpoint['model']

        new_state_dict = model.state_dict()
        for k, v in state_dict.items():
            if bool(len(v.shape)) and v.shape[0] ==255:
                continue
            if 'module.' in k:
                namekey = k[7:] # remove `module.`
                new_state_dict[namekey] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

        # for i, (name, p) in enumerate(model.named_parameters()):
        #     p.requires_grad = True if (p.shape[0] == 24) else False
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=.9)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device).train()

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)
    # Start training
    t0 = time.time()
    model_info(model)
    n_burnin = min(round(len(dataloader) / 5 + 1), 1000)  # burn-in batches
    for epoch in range(epochs):
        model.train()
        epoch += start_epoch

        print(('\n%8s%12s' + '%10s' * 7) % (
            'Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler (automatic)
        # scheduler.step()

        # Update scheduler (manual)
        if epoch > 250:
            lr = lr0 / 10
        else:
            lr = lr0
        for x in optimizer.param_groups:
            x['lr'] = lr

        # Freeze darknet53.conv.74 for first epoch
        if freeze_backbone and (epoch == 0):
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        optimizer.zero_grad()
        rloss = defaultdict(float)
        for i, (imgs, targets, _, _) in enumerate(dataloader):
            targets = targets.to(device)
            nT = targets.shape[0]
            if nT == 0:  # if no targets continue
                continue

            # SGD burn-in
            if (epoch == 0) and (i <= n_burnin):
                lr = lr0 * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            pred = model(imgs.to(device))

            # Build targets
            target_list = build_targets(model, targets, pred)

            # Run model
            pred = model(imgs.to(device))

            # Build targets
            target_list = build_targets(model, targets, pred)

            # Compute loss
            # loss = model(imgs.to(device), targets, var=var)
            loss, loss_dict = compute_loss(pred, target_list)

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1
            for key, val in loss_dict.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            s = ('%8s%12s' + '%10.5g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, len(dataloader) - 1),
                rloss['xy'], rloss['wh'], rloss['conf'],
                rloss['cls'], rloss['total'],
                nT, time.time() - t0)
            t0 = time.time()
            
            print(s)

        # Update best loss
        if rloss['total'] < best_loss:
            best_loss = rloss['total']
            checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, best)

        # Save latest checkpoint

        # Save backup weights every 5 epochs (optional)
        if (epoch > 0) & (epoch % 20 == 0):
            checkpoint = {'epoch': epoch,
                'best_loss': best_loss,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, latest)

            # Calculate mAP
            with torch.no_grad():
                temp_model = Darknet(cfg, img_size)
                new_state_dict = OrderedDict()
                for k, v in model.state_dict().items():
                    if 'module.' in k:
                        namekey = k[7:] # remove `module.`
                        new_state_dict[namekey] = v
                    else:
                        new_state_dict[k] = v
                temp_model.load_state_dict(new_state_dict)
                mAP, R, P = test.test(temp_model, data_cfg, batch_size=batch_size, img_size=img_size)

                # Write epoch results
                with open('results.txt', 'a') as file:
                    file.write(s + '%11.3g' * 3 % (mAP, P, R) + '\n')
    #writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=20, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--cfg', type=str, default='cfg/xy_yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/xy.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--resume', action='store_true',default=True, help='resume training flag')
    parser.add_argument('--var', type=float, default=0, help='test variable')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()
    
    train(
        opt.cfg,
        opt.data_cfg,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        multi_scale=opt.multi_scale,
    )
