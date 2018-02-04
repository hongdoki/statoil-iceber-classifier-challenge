import numpy as np
import random
import torch
import numbers

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for transform in self.transforms:
            img = transform(img)
        return img

class HFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return np.ascontiguousarray(np.fliplr(img))
        return img

class VFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return np.ascontiguousarray(np.flipud(img))
        return img

class ToTensor(object):
    def __init__(self, format='NCHW'):
        self.format = format

    def __call__(self, img):
        tensor = torch.from_numpy(img)
        tensor = tensor.contiguous()
        tensor = tensor.transpose(0, 1).transpose(0,2)
        return tensor

class FiveCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return five_crop(img, self.size)
        
class RandomFiveCrop(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        return random.choice(five_crop(img, self.size))

def five_crop(img, size):

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size)==2, "Please provide only two dimensions (h, w) for size"
    h, w = img.shape[:2]
    crop_h, crop_w = size
    if crop_h > h or crop_w > w:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size, (h, w)))
    
    tl = img[:crop_h, :crop_w]
    tr = img[:crop_h, -crop_w:]
    bl = img[-crop_h:, :crop_w]
    br = img[-crop_h:, -crop_w:]
    center = img[h//2 - crop_h//2 : h//2 - crop_h//2 + crop_h,
                 w//2 - crop_w//2 : w//2 - crop_w//2 + crop_w]
    return (tl, tr, bl, br, center)

