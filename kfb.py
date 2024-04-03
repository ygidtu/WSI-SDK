#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import io

import numpy as np
import cv2

from openslide import AbstractSlide, _OpenSlideMap
from PIL import Image
from tqdm import tqdm

import utils


class kfbRef:
    img_count = 0


class TSlide(AbstractSlide):
    def __init__(self, filename):
        AbstractSlide.__init__(self)
        self.__filename = filename
        self._osr = utils.kfbslide_open(filename)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.__filename)

    @classmethod
    def detect_format(cls, filename):
        return utils.detect_vendor(filename)

    def close(self):
        utils.kfbslide_close(self._osr)

    @property
    def level_count(self):
        return utils.kfbslide_get_level_count(self._osr)

    @property
    def level_dimensions(self):
        return tuple(utils.kfbslide_get_level_dimensions(self._osr, i)
                     for i in range(self.level_count))

    @property
    def level_downsamples(self):
        return tuple(utils.kfbslide_get_level_downsample(self._osr, i)
                     for i in range(self.level_count))

    @property
    def properties(self):
        return _KfbPropertyMap(self._osr)

    @property
    def associated_images(self):
        return _AssociatedImageMap(self._osr)

    def get_best_level_for_downsample(self, downsample):
        return utils.kfbslide_get_best_level_for_downsample(self._osr, downsample)

    def read_region(self, location, level, size):
        # import pdb

        x = int(location[0])
        y = int(location[1])
        width = int(size[0])
        height = int(size[1])
        img_index = kfbRef.img_count
        kfbRef.img_count += 1

        return Image.open(io.BytesIO(utils.kfbslide_read_roi_region(self._osr, level, x, y, width, height)))

    def read_whole_image(self, level = 0):
        patch_size = 4000
        [x, y] = self.level_dimensions[level]
        # print("Resolution --> {}, {}".format(x, y))
        image = np.zeros([y, x, 3], np.uint8)
        x_range = list(range(0, x, patch_size))
        y_range = list(range(0, y, patch_size))
        for i in tqdm(x_range):
            for j in y_range:
                x_size = y_size = patch_size
                if i == max(x_range):
                    x_size = x - i
                if j == max(y_range):
                    y_size = y - j
                patch = np.array(self.read_region((i, j), level, (x_size, y_size)))
                if patch.shape[2] == 4:
                    r, g, b, a = cv2.split(patch)
                    patch = cv2.merge([r, g, b])
                image[j:j + y_size, i:i + x_size, :] = patch
        return image

    def get_thumbnail(self, size):
        """Return a PIL.Image containing an RGB thumbnail of the image.

        size:     the maximum size of the thumbnail."""
        if isinstance(size, int) or isinstance(size, float):
            size = [size, size]

        img = Image.fromarray(self.read_whole_image(0))
        return img.resize(size)


class _KfbPropertyMap(_OpenSlideMap):
    def _keys(self):
        return utils.kfbslide_property_names(self._osr)

    def __getitem__(self, key):
        v = utils.kfbslide_property_value(self._osr, key)
        if v is None:
            raise KeyError()
        return v


class _AssociatedImageMap(_OpenSlideMap):
    def _keys(self):
        return utils.kfbslide_get_associated_image_names(self._osr)

    def __getitem__(self, key):
        if key not in self._keys():
            raise KeyError()
        return utils.kfbslide_read_associated_image(self._osr, key)


if __name__ == '__main__':
    pass
