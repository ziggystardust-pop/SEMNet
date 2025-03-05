from test_dataset import TestMudrockDataset
import os
import time
import datetime
import torch
from train_utils import evaluate
from src import SEMNet
import transforms as T
from torch.utils.data import DataLoader

class SegmentationPresetEval:
    def __init__(self,target_size=(480,480), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    def __call__(self, img, target):
        return self.transforms(img, target)
def get_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return SegmentationPresetEval(mean=mean,target_size=(256,256), std=std)


def create_model(args,num_classes):
    if(args.model == "SEMNet"):
        model = SEMNet(in_channels=3, num_classes=num_classes)
    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = args.num_classes + 1
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    num_workers = 16
    test_dataset = TestMudrockDataset(args.data_path,
                               transforms=get_transform(mean=mean, std=std))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=test_dataset.collate_fn)
    model = create_model(args,num_classes=num_classes)
    model.to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model from {args.resume}")
    start_time = time.time()
    confmat, test_dice, test_loss = evaluate(model, test_loader, device=device, num_classes=num_classes,ds=args.ds)
    val_info = str(confmat)
    print(val_info)
    print(f"test Dice Coefficient: {test_dice:.3f}")
    print(f"test Loss: {test_loss:.4f}")
    test_mean_iou = confmat.mean_iou()
    results_file = os.path.join("test_log", f"{args.model}_{args.dataname}_{args.num_classes}_test_results.txt")
    with open(results_file, "w") as f:
        test_info = f"test Results:\n" \
                    f"Mean IoU: {test_mean_iou:.4f}\n" \
                    f"Dice Coefficient: {test_dice:.3f}\n" \
                    f"Loss: {test_loss:.4f}\n"
        f.write(test_info + val_info + "\n\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    """
    mudnet rock
    """
    parser.add_argument("--data-path", default="/dataset/data", help="DRIVE root")
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--model", default="SEMNet")
    parser.add_argument("--dataname", default="mudrock")
    parser.add_argument('--ds',type=int, default=0)
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
