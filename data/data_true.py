from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from skimage import io, color, transform
from skimage.color import rgb2gray
from utils import img_segments_only, resize


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        img = sample.transpose((2, 0, 1))
        return torch.from_numpy(img)

class Rescale(object):
    """Rescale the image in a sample to a given size."""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.shape[:2]
        if isinstance(self.output_size, int):
            new_h, new_w = (self.output_size * h / w, self.output_size) if h > w else (self.output_size, self.output_size * w / h)
        else:
            new_h, new_w = self.output_size
        return transform.resize(sample, (int(new_h), int(new_w)))


class MyData_test(Dataset):
    def __init__(self, target_path, ref_path, slic_target, transform=None, target_transfom=None, slic=True, size=224,
                 color_space='lab'):
        self.target_path = target_path
        self.slic = slic
        self.transform = transform
        self.target_transform = target_transfom
        self.size = size
        self.color_space = color_space
        self.slic_target = slic_target
        self.ref_path = ref_path

    def __getitem__(self, index):

        x_real = rgb2gray(io.imread('./samples/target/' + self.target_path[index], pilmode='RGB'))
        x = rgb2gray(resize(io.imread('./samples/target/' + self.target_path[index], pilmode='RGB'),
                            (224, 224)))  # Reading target images in RGB
        ref_real = io.imread('./samples/ref/' + self.ref_path[index], pilmode='RGB')
        ref = resize(io.imread('./samples/ref/' + self.ref_path[index], pilmode='RGB'),
                     (224, 224))  # Reading ref images in RGB

        if self.color_space == 'lab':
            if np.ndim(x) == 3:

                x_luminance_classic_real = (x_real[:, :, 0])
                x_luminance_classic = (x[:, :, 0])

            else:
                x_luminance_classic_real = (x_real)
                x_luminance_classic = (x)
                x = x[:, :, np.newaxis]
                x_real = x_real[:, :, np.newaxis]

            ref_new_color = color.rgb2lab(ref)
            ref_luminance_classic = (ref_new_color[:, :, 0] / 100.0)
            ref_chroma = ref_new_color[:, :, 1:] / 127.0

            # Luminance remapping
            x_luminance_map = (np.std(ref_luminance_classic) / np.std(x_luminance_classic)) * (
                        x_luminance_classic - np.mean(x_luminance_classic)) + np.mean(ref_luminance_classic)

        # Calculating superpixel label map for target and reference images (Grayscale)

        # The following operations are assumed to be similar, could be further optimized based on `img_segments_only` implementation
        target_slic = [img_segments_only(x_luminance_classic, 2 ** i, int(self.size / (2 ** i))) for i in range(3)]
        ref_slic = [img_segments_only(ref_luminance_classic, 2 ** i, int(self.size / (2 ** i))) for i in range(3)]

        # Applying transformation (To tensor) and replicating tensor for gray scale images
        if self.target_transform:
            transforms = [self.target_transform(np.expand_dims(slice, axis=2)) for slice in
                          target_slic + ref_slic]
            target_slic_all = transforms[:3]
            ref_slic_all = transforms[3:6]

            x = self.target_transform(x)
            ref_real = self.target_transform(ref_real)
            ref = self.target_transform(ref)
            x_luminance_map = self.target_transform(x_luminance_map[:, :, np.newaxis])
            x_luminance_classic_real = self.target_transform(x_luminance_classic_real[:, :, np.newaxis])
            x_luminance_classic_real_rep = torch.cat(
                (x_luminance_classic_real.float(), x_luminance_classic_real.float(), x_luminance_classic_real.float()),
                dim=0)
            luminance_replicate_map = torch.cat(
                (x_luminance_map.float(), x_luminance_map.float(), x_luminance_map.float()),
                dim=0)
            ref_chroma = self.target_transform(ref_chroma)
            ref_luminance_replicate = torch.cat([self.target_transform(ref_luminance_classic[:, :, np.newaxis])] * 3,
                                                dim=0)

        # Output: x: target image rgb, ref: reference image rdb, luminance_replicate: target grayscale image replicate, ref_luminance_replicate: reference grayscale image replicate
        # labels_torch: label map target image, labels_ref_torch: label map reference image, x_luminance: target grayscale image, ref_luminance: reference grayscale image.
        #
        return x, x_luminance_classic_real_rep, ref, ref_luminance_replicate, target_slic_all, ref_slic_all, ref_chroma, luminance_replicate_map, x_luminance_classic_real_rep, ref_real

    def __len__(self):
        return len(self.target_path)








