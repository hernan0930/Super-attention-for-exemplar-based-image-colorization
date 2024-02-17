import os
import torch
from utils import imagenet_norm
from torchvision.utils import save_image

def testing_color(loader_val, out_dir, model_color, model_percep, device):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with torch.no_grad():

        model_color.eval()
        for idx, (img_target_gray, img_target_gray_real, ref_rgb, ref_gray, target_slic, ref_slic_all, img_ref_ab,
        img_gray_map, gray_real, ref_real) in enumerate(loader_val):

            # Target data
            img_gray_map = (img_gray_map).to(device=device, dtype=torch.float)
            img_target_gray = (img_target_gray).to(device=device, dtype=torch.float)
            gray_real = gray_real.to(device=device, dtype=torch.float)
            target_slic = target_slic

            # Loading references
            ref_rgb_torch = ref_rgb.to(device=device, dtype=torch.float)
            img_ref_gray = (ref_gray).to(device=device, dtype=torch.float)
            ref_slic_all = ref_slic_all

            # VGG19 normalization
            img_ref_rgb_norm = imagenet_norm(ref_rgb_torch, device)

            # VGG19 normalization


            feat1_pred, feat2_pred, feat3_pred, _, _ = model_percep(img_ref_rgb_norm)


            ab_pred, pred_Lab_torch, pred_RGB_torch = model_color(img_ref_gray,
                                                                  img_target_gray,
                                                                  target_slic,
                                                                  ref_slic_all,
                                                                  img_gray_map,
                                                                  gray_real,
                                                                  feat1_pred, feat2_pred, feat3_pred,
                                                                  device)
            save_image(pred_RGB_torch,
                            out_dir + str(idx) + '_pred.png',
                           normalize=True)

            print('Image saved in', out_dir + str(idx) + '_pred.png')



