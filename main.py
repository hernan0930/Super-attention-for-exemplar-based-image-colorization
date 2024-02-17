import argparse
from data.data_true import *
import testing.testing
from models.super_skip_unet import *
from models.model_perceptual import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def testing_mode(path_target, path_ref, out_dir, device):
    datasett_val = MyData_test(target_path=path_target, ref_path=path_ref, slic_target=None,
                               transform=None, target_transfom=ToTensor())

    loader_val = DataLoader(datasett_val, batch_size=1, shuffle=False, pin_memory=True)

    # Load models
    PATH_model = './save_models/our_l1_lpips_hist_encod_max_lum_map_multiref_w_conv_ChromaGan_epoch23/checkpoint.pt'

    # Load model and initializing VGG 19 weights and bias
    model2 = gen_color_stride_vgg16(dim=2)
    model2.load_state_dict(torch.load(PATH_model, map_location=device)['state_dict'])
    model2.to(device=device)
    model2.eval()

    model_percep = percep_vgg19_bn().to(device=device)
    model_percep.eval()

    for param_percep in model_percep.parameters():
        param_percep.requires_grad = False

    for param_color in model2.parameters():
        param_color.requires_grad = False

    testing.testing.testing_color(loader_val, out_dir,
                                  model2,
                                  model_percep,
                                  device)


#############################################################################
#                                Main Block                                 #
#############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True, default='./samples/target/', help='Directory of target images')
    parser.add_argument('--ref_dir', type=str, required=True, default='./samples/ref/', help='Directory of reference images')
    parser.add_argument('--out_dir', type=str, required=True, default='./samples/results/', help='Directory for resulting images')
    args = parser.parse_args()

    # Loading images in folder
    target_images = list_image_files(args.target_dir)  # Lists target image files.
    reference_images = list_image_files(args.ref_dir)  # Lists reference image files.

    # Testing function
    testing_mode(target_images, reference_images, args.out_dir, device)


