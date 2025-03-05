

import os
from PIL import Image
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "train" if train else "val"
        self.data_root = os.path.join(root, "ImageSets", "Segmentation")
        self.transforms = transforms
        if self.flag == "train":
            txt_file = os.path.join(self.data_root, "train.txt")
        else:
            txt_file = os.path.join(self.data_root, "val.txt")
        assert os.path.exists(txt_file), f"File '{txt_file}' does not exist."
        with open(txt_file, "r") as f:
            img_names = [line.strip() for line in f.readlines()]

        img_dir = os.path.join(root, "JPEGImages")
        mask_dir = os.path.join(root, "SegmentationClass")

        self.img_list = [os.path.join(img_dir, name + ".jpg") for name in img_names]
        self.mask_list = [os.path.join(mask_dir, name + ".png") for name in img_names]

        for mask_path in self.mask_list:
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"File '{mask_path}' does not exist.")

    def __getitem__(self, idx):

        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.mask_list[idx])

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)


        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

def cat_list(images, fill_value=0):
        
    max_size = tuple(max(s) for s in zip(*[img.size() for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

