# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import numpy as np

from plsc.data import preprocess as transforms


class MAERandCropImage(transforms.RandCropImage):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    
    def __call__(self, img):
        size = self.size

        img_h, img_w = img.shape[:2]

        target_area = img_w * img_h * np.random.uniform(*self.scale)
        log_ratio = tuple(math.log(x) for x in self.ratio)
        aspect_ratio = math.exp(np.random.uniform(*log_ratio))
        
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        
        w = min(w, img_w)
        h = min(h, img_h)

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        img = img[j:j + h, i:i + w, :]

        return self._resize_func(img, size)