import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFilter


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class CutMix:
    def __init__(self, alpha=1.0, probability=0.5):

        self.alpha = alpha
        self.probability = probability

    def __call__(self, image, mask):

        if random.random() < self.probability:
            is_pil_image = isinstance(image, Image.Image)
            is_tensor = isinstance(image, torch.Tensor)

            if is_tensor:
                image = F.to_pil_image(image)
                mask = F.to_pil_image(mask.float())
            elif not is_pil_image:
                raise TypeError("Input image must be a PIL Image or Tensor.")

            image_tensor = F.to_tensor(image)
            mask_tensor = F.to_tensor(mask).squeeze(0).long()

            shuffled_image = image_tensor
            shuffled_mask = mask_tensor

            lam = np.random.beta(self.alpha, self.alpha)
            cx = np.random.uniform(0, image_tensor.size(-1))
            cy = np.random.uniform(0, image_tensor.size(-2))
            w = image_tensor.size(-1) * np.sqrt(1 - lam)
            h = image_tensor.size(-2) * np.sqrt(1 - lam)
            x1 = int(np.round(max(cx - w / 2, 0)))
            y1 = int(np.round(max(cy - h / 2, 0)))
            x2 = int(np.round(min(cx + w / 2, image_tensor.size(-1))))
            y2 = int(np.round(min(cy + h / 2, image_tensor.size(-2))))

            if x2 > x1 and y2 > y1:
                mixed_image = image_tensor.clone()
                mixed_image[:, y1:y2, x1:x2] = shuffled_image[:, y1:y2, x1:x2]

                mixed_mask = mask_tensor.clone()
                mixed_mask[y1:y2, x1:x2] = shuffled_mask[y1:y2, x1:x2]

                mixed_image = F.to_pil_image(mixed_image)
                mixed_mask = F.to_pil_image(mixed_mask.float())

                return mixed_image, mixed_mask

        return image, mask

class RandomTranslate(object):
    def __init__(self, max_shift=10, probability=0.25):
        self.max_shift = max_shift
        self.probability = probability

    def __call__(self, image, target):
        if random.random() < self.probability:
            x_shift = random.randint(-self.max_shift, self.max_shift)
            y_shift = random.randint(-self.max_shift, self.max_shift)

            image = F.affine(
                image,
                angle=0,
                translate=(x_shift, y_shift),
                scale=1.0,
                shear=0,
                interpolation=F.InterpolationMode.BILINEAR,
            )
            target = F.affine(
                target,
                angle=0,
                translate=(x_shift, y_shift),
                scale=1.0,
                shear=0,
                interpolation=F.InterpolationMode.NEAREST,
            )
        return image, target
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        noise = torch.randn_like(F.to_tensor(image)) * self.std + self.mean
        image = F.to_tensor(image) + noise
        image = torch.clamp(image, 0, 1)
        return F.to_pil_image(image), target

class AddRingArtifact(object):
    def __init__(self, radius_range=(5, 15), intensity=100):
        self.radius_range = radius_range
        self.intensity = intensity

    def __call__(self, image, target):
        image_np = np.array(image.convert("L"))
        h, w = image_np.shape
        radius = random.randint(*self.radius_range)
        
        ring = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(ring)
        center = (w // 2, h // 2)
        draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], outline=self.intensity)
        
        ring_blurred = ring.filter(ImageFilter.GaussianBlur(radius=3))
        image_with_artifact = Image.fromarray(np.clip(image_np + np.array(ring_blurred), 0, 255).astype(np.uint8)).convert("RGB")
        return image_with_artifact, target

class RandomRotate(object):
    def __init__(self, angle_range=(-30, 30)):
        self.angle_range = angle_range

    def __call__(self, image, target):
        angle = random.uniform(*self.angle_range)
        image = F.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
        target = F.rotate(target, angle, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target
class RandomRotate90(object):
    def __init__(self, rotate_prob = 0.5):

        self.rotate_prob = rotate_prob

    def __call__(self, image, target):

        if random.random() < self.rotate_prob:
            rotations = [lambda img: F.rotate(img, 90), lambda img: F.rotate(img, 270)]
            rotation = random.choice(rotations)
            image = rotation(image)
            target = rotation(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, image, target):
        resize = T.Resize(self.size)
        image = resize(image)
        target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)  # 这里使用 F.resize
        return image, target
