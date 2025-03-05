import os
import time
import datetime
import torch
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset
import transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from src import SEMNet
class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)
        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)
    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self,target_size=(480,480), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 300
    crop_size = 256
    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean,target_size=(256,256), std=std)


def create_model(args,num_classes):
  if(args.model == "SEMNet"):
    model = SEMNet(in_channels=3, num_classes=num_classes)

  return model


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes + 1
    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    results_file = "res/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(results_file, "a") as f:
        model_info = f"[model: {args.model}]\n" \
                         f"data: {args.data_path}\n" \
                         f"num_classes: {args.num_classes + 1}\n" \
                         f"num_classes: {args.dataname}\n"

        f.write(model_info+"\n\n")


    train_dataset = DriveDataset(args.data_path,
                                        train=True,
                                        transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                                train=False,
                                transforms=get_transform(train=False, mean=mean, std=std))
    num_workers = 16
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(args,num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        params=params_to_optimize,
        lr=args.lr,
        betas=(args.momentum, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{args.model}_{args.dataname}_{args.num_classes}_{timestamp}"
    log_dir = os.path.join("logs", experiment_name)
    writer = SummaryWriter(log_dir)
    best_miou = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler,ds=args.ds)
        print("train Loss:",mean_loss)

        confmat, val_dice,val_loss = evaluate(model, val_loader, device=device, num_classes=num_classes,ds=args.ds)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {val_dice:.3f}")
        val_mean_iou = confmat.mean_iou()
        writer.add_scalar('Loss/train', mean_loss, epoch)       # 记录训练损失
        writer.add_scalar('Loss/val', val_loss, epoch)          # 记录验证损失
        writer.add_scalar('Dice/val', val_dice, epoch)          # 记录验证集的Dice系数
        writer.add_scalar('Mean_IOU/val', val_mean_iou, epoch)      # 记录验证集的Mean IoU
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {val_dice:.3f}\n" \
                         f"Validation Loss: {val_loss:.4f}\n"
            f.write(train_info + val_info + "\n\n")
        if args.save_best is True:
            if best_miou < val_mean_iou:
                best_miou = val_mean_iou
                save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
                if args.amp:
                    save_file["scaler"] = scaler.state_dict()
                save_dir = "./save_weights/{}/{}".format(args.dataname, args.model)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(save_file,"save_weights/{}/{}/best.pth".format(args.dataname,args.model))
                print("best miou:",best_miou)
                print("save pth:",save_dir)
            else:
                continue
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--data-path", default="/dataset/data", help="DRIVE root")
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--model", default="SEMNet")
    parser.add_argument("--dataname", default="mudrock")
    parser.add_argument('--ds', default=0,type=int,help="deepsupervision")
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=270, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
