import os
from skimage.transform import rescale, resize
from skimage.util import img_as_float
from skimage.segmentation import slic
import torch


def list_image_files(directory):
    supported_extensions = ['.jpg', '.jpeg', '.png']

    # Extracts the numeric part of the filename and its extension for sorting.
    def file_sort_key(filename):
        # Extract the base name without extension
        basename = os.path.splitext(filename)[0]
        # Extract the numeric part of the filename
        try:
            numeric_part = int(basename)
        except ValueError:
            # If not a clear numeric value, default to a high number to sort these last
            numeric_part = float('inf')
        # Get the extension's priority (index in the supported_extensions list)
        extension_priority = supported_extensions.index(os.path.splitext(filename)[1].lower())
        return numeric_part, extension_priority

    # Filter and sort files based on the defined sorting key.
    files = [f for f in os.listdir(directory) if
             os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in supported_extensions]
    files.sort(key=file_sort_key)

    return files


def img_segments_only(img_grays, div_resize, num_max_seg):
    """
    :param img_grays: Target image
    :param div_resize: Resizing factor
    :return: Superpixel's label map for the target and reference images.
    """

    # Gray scale or RGB image
    if img_grays.ndim == 2:
        image_gray_2 = img_as_float(resize(img_grays, (img_grays[0].size / div_resize, img_grays[1].size / div_resize)))
        segment_gray_2 = slic(image_gray_2, n_segments=int(num_max_seg), sigma=0, compactness=0.1, channel_axis=None)

    else:
        image_gray_2 = img_as_float(resize(img_grays, (len(img_grays[0]) / div_resize, len(img_grays[1]) / div_resize)))
        segment_gray_2 = slic(image_gray_2, n_segments=int(num_max_seg), sigma=0, compactness=9)


    return segment_gray_2


def imagenet_norm(input, device):
    # VGG19 normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device=device, dtype=torch.float)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device=device, dtype=torch.float)
    out = (input - mean.view((1, 3, 1, 1))) / std.view((1, 3, 1, 1))
    return out

