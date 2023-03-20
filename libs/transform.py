import random
import os

import numpy as np
import cv2
import torch

from torchvision.transforms import functional as F

from libs.utils import (
    generate_shiftscalerotate_matrix,
    distort_hsv,
    distort_noise,
    distort_smooth
)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)

        return img, target

    def __repr__(self):
        format_str = self.__class__.__name__ + '('
        for t in self.transforms:
            format_str += '\n'
            format_str += f'    {t}'
        format_str += '\n)'

        return format_str

class Grayscalize:
    def __init__(self, flag=False):
        self.flag = flag

    def __call__(self, img, target):
        if self.flag:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.merge([img,img,img]) # three channels by duplication
        return img, target

class Resize:
    def __init__(self, dst_width, dst_height, dst_K):
        self.dst_width = dst_width
        self.dst_height = dst_height
        self.dst_K = dst_K

    def __call__(self, img, target):
        M = np.matmul(self.dst_K, np.linalg.inv(target.K))
        # 
        img = cv2.warpAffine(img, M[:2], (self.dst_width, self.dst_height), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
        target = target.transform(M, self.dst_K, self.dst_width, self.dst_height)
        return img, target

class RandomShiftScaleRotate:
    def __init__(self, shift_limit, scale_limit, rotate_limit, dst_width, dst_height, dst_K):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        # 
        self.dst_width = dst_width
        self.dst_height = dst_height
        self.dst_K = dst_K

    def __call__(self, img, target):
        if (self.shift_limit + self.scale_limit + self.rotate_limit) > 0.01:
            M = generate_shiftscalerotate_matrix(
                    self.shift_limit, self.scale_limit, self.rotate_limit, 
                    self.dst_width, self.dst_height
                )
            img = cv2.warpAffine(img, M[:2], (self.dst_width, self.dst_height), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
            target = target.transform(M, self.dst_K, self.dst_width, self.dst_height)
        return img, target

class RandomHSV:
    def __init__(self, h_ratio, s_ratio, v_ratio):
        self.h_ratio = h_ratio
        self.s_ratio = s_ratio
        self.v_ratio = v_ratio
    def __call__(self, img, target):
        if (self.h_ratio + self.s_ratio + self.v_ratio) > 0.01:
            img = distort_hsv(img, self.h_ratio, self.s_ratio, self.v_ratio)
        return img, target

class RandomNoise:
    def __init__(self, noise_ratio):
        self.noise_ratio = noise_ratio
    def __call__(self, img, target):
        if self.noise_ratio > 0.01:
            img = distort_noise(img, self.noise_ratio)
        return img, target

class RandomSmooth:
    def __init__(self, max_ksize = 5):
        self.max_ksize = max_ksize
        self.ks_candidates = []
        i = 1
        while i <= self.max_ksize:
            self.ks_candidates.append(i)
            i += 2
    def __call__(self, img, target):
        if self.max_ksize > 1:
            ks = random.choice(self.ks_candidates)
            img = cv2.blur(img, (ks, ks))
        return img, target

class RandomPencilSharpen:
    def __init__(self, ratio=0.5):
        self.sharpen_ratio = ratio

    def __call__(self, img, target):
        if random.random() < self.sharpen_ratio:
            ks_candidates = [5, 7, 9, 11]
            ks = random.choice(ks_candidates)
            img_s = cv2.blur(img, (ks, ks))
            if random.random() < 0.5:
                edge = img / (img_s.astype(np.float32) + 0.01)
            else:
                edge = img - img_s.astype(np.float32)
            edge = cv2.normalize(edge, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
            alpha = random.uniform(0.5, 0.95)
            img = img * (1-alpha) + edge * alpha
            img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        return img, target

class RandomBackground:
    def __init__(self, background_dir):
        self.background_files = []
        try:
            if os.path.isdir(background_dir):
                self.background_files = [
                    background_dir + f 
                    for f in os.listdir(background_dir) if f.endswith('.png') or f.endswith('.jpg')
                    ]
        except:
            # can not read background directory, remains empty
            pass
        print("Number of background images: %d" % len(self.background_files))

    def __call__(self, img, target):
        if np.random.rand() < 0.5:
            if len(self.background_files) > 0:
                if img.shape[2] == 4:
                    img = self.merge_background_alpha(img, self.get_a_random_background())
                else:
                    img = self.merge_background_mask(img, self.get_a_random_background(), target.mask)
            else:
                img = img[:,:,0:3]
        else:
            img = img[:,:,0:3]
        return img, target
    
    def get_a_random_background(self):
        backImg = None
        while backImg is None:
            backIdx = random.randint(0, len(self.background_files) - 1)
            img_path = self.background_files[backIdx]
            try:
                backImg = cv2.imread(img_path)
                if backImg is None:
                    raise RuntimeError('load image error')
            except:
                print('Error in loading background image: %s' % img_path)
                backImg = None
        return backImg

    def merge_background_alpha(self, foreImg, backImg):
        assert(foreImg.shape[2] == 4)
        forergb = foreImg[:, :, :3]
        alpha = foreImg[:, :, 3] / 255.0
        if forergb.shape != backImg.shape:
            backImg = cv2.resize(backImg, (foreImg.shape[1], foreImg.shape[0]))
        alpha = np.repeat(alpha, 3).reshape(foreImg.shape[0], foreImg.shape[1], 3)
        mergedImg = np.uint8(backImg * (1 - alpha) + forergb * alpha)
        # backImg[alpha > 128] = forergb[alpha > 128]
        return mergedImg

    def merge_background_mask(self, foreImg, backImg, maskImg):
        forergb = foreImg[:, :, :3]
        if forergb.shape != backImg.shape:
            backImg = cv2.resize(backImg, (foreImg.shape[1], foreImg.shape[0]))
        alpha = np.ones((foreImg.shape[0], foreImg.shape[1], 3), np.float32)
        alpha[maskImg == 0] = 0
        mergedImg = np.uint8(backImg * (1 - alpha) + forergb * alpha)
        # backImg[alpha > 128] = forergb[alpha > 128]
        return mergedImg

class RandomBackground_Chance:
    def __init__(self, background_dir):
        self.background_files = []
        try:
            VOC_root = background_dir  # path to "VOCdevkit/VOC2012"
            VOC_image_set_dir = os.path.join(VOC_root, "ImageSets/Main")
            VOC_bg_list_path = os.path.join(VOC_image_set_dir, "diningtable_trainval.txt")
            # self.background_files = []
            with open(VOC_bg_list_path, "r") as f:
                    VOC_bg_list = [
                        line.strip("\r\n").split()[0] for line in f.readlines() if line.strip("\r\n").split()[1] == "1"
                    ]
            self.background_files = [os.path.join(VOC_root, "JPEGImages/{}.jpg".format(bg_idx)) for bg_idx in VOC_bg_list]
        except:
            pass
        print("Number of background images: %d" % len(self.background_files))

    def __call__(self, img, target):
        if np.random.rand() < 0.5:
            if len(self.background_files) > 0:
                if img.shape[2] == 4:
                    img = self.merge_background_alpha(img, self.get_a_random_background())
                else:
                    img = self.merge_background_mask(img, self.get_a_random_background(), target.mask)
            else:
                img = img[:,:,0:3]
        else:
            img = img[:,:,0:3]
        return img, target
    
    def get_a_random_background(self):
        backImg = None
        while backImg is None:
            backIdx = random.randint(0, len(self.background_files) - 1)
            img_path = self.background_files[backIdx]
            try:
                backImg = cv2.imread(img_path)
                if backImg is None:
                    raise RuntimeError('load image error')
            except:
                print('Error in loading background image: %s' % img_path)
                backImg = None
        return backImg

    def merge_background_alpha(self, foreImg, backImg):
        assert(foreImg.shape[2] == 4)
        forergb = foreImg[:, :, :3]
        alpha = foreImg[:, :, 3] / 255.0
        if forergb.shape != backImg.shape:
            backImg = cv2.resize(backImg, (foreImg.shape[1], foreImg.shape[0]))
        alpha = np.repeat(alpha, 3).reshape(foreImg.shape[0], foreImg.shape[1], 3)
        mergedImg = np.uint8(backImg * (1 - alpha) + forergb * alpha)
        # backImg[alpha > 128] = forergb[alpha > 128]
        return mergedImg

    def merge_background_mask(self, foreImg, backImg, maskImg):
        forergb = foreImg[:, :, :3]
        if forergb.shape != backImg.shape:
            backImg = cv2.resize(backImg, (foreImg.shape[1], foreImg.shape[0]))
        alpha = np.ones((foreImg.shape[0], foreImg.shape[1], 3), np.float32)
        alpha[maskImg == 0] = 0
        mergedImg = np.uint8(backImg * (1 - alpha) + forergb * alpha)
        # backImg[alpha > 128] = forergb[alpha > 128]
        return mergedImg

class RandomOcclusion:
    """
    randomly erasing holes
    ref: https://arxiv.org/abs/1708.04896
    """
    def __init__(self, prob = 0):
        self.prob = prob

    def __call__(self, img, target):
        if self.prob > 0:
            height, width, channels = img.shape
            bboxes = target.to_visible_boxlist()
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = [int(v) for v in bboxes[i].bbox]
                bw = int(x2-x1)
                bh = int(y2-y1)
                if random.uniform(0, 1) <= self.prob and bw > 2 and bh > 2:
                    bb_size = bw*bh
                    size = random.uniform(0.02, 0.7) * bb_size
                    ratio = random.uniform(0.5, 2.0)
                    ew = int(np.sqrt(size * ratio))
                    eh = int(np.sqrt(size / ratio))
                    ecx = random.uniform(x1, x2)
                    ecy = random.uniform(y1, y2)
                    esx = int(np.clip((ecx - ew/2 + 0.5), 0, width-1))
                    esy = int(np.clip((ecy - eh/2 + 0.5), 0, height-1))
                    eex = int(np.clip((ecx + ew/2 + 0.5), 0, width-1))
                    eey = int(np.clip((ecy + eh/2 + 0.5), 0, height-1))
                    targetshape = img[esy:eey, esx:eex, :].shape
                    img[esy:eey, esx:eex, :] = np.random.randint(256, size=targetshape)
                    if channels == 4:
                        img[esy:eey, esx:eex, 3] = 255
                    target.mask[esy:eey, esx:eex] = -1
        return img, target

class ToTensor:
    def __call__(self, img, target):
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        target = target.to_tensor()
        return img, target
    
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        img = img - np.array(self.mean).reshape(1,1,3)
        img = img / np.array(self.std).reshape(1,1,3)
        return img, target
