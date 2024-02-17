from torch.nn import init
import torch.nn as nn
from torch_scatter import scatter_max
import kornia
from utils import *

#############################################################################
#   Functions and Model Block                                             #
#############################################################################

def init_layer(layer, init_type='xavier', gain=0.02):

    """
        Weights initialization for convolutional and linear layers.
        :param layer: Input model
        :param init_type: 'xavier', 'normal', 'kaiming', 'orthogonal'
        :param gain: Hyperparameter for xavier init
        :return:
    """
    classname = layer.__class__.__name__
    torch.manual_seed(150)
    if hasattr(layer, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            init.normal_(layer.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            init.xavier_normal_(layer.weight.data, gain=gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(layer.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init.orthogonal_(layer.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.constant_(layer.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(layer.weight.data, 1.0, gain)
        init.constant_(layer.bias.data, 0.0)
    return layer


"""
Encoded superpixel features
"""

class Superfeatures(nn.Module):

    def __init__(self):

        super(Superfeatures, self).__init__()

    def forward(self, input_features_in, label_mask, device):
        """
        :param input_features: Feature tensor [B, C, H, W]
        :param label_mask: int label mask of size [B, C, H, W]
        :return cat_encoded: Staked array's of masked tensor of size [B, C, N_max = M_max]
        """

        batch_size, n_channels = input_features_in.shape[:2]
        sp_maps_expand = label_mask.expand(-1, n_channels, -1, -1).to(device)
        # super_features, _ = scatter_max(input_features_in.flatten(-2).to(device), sp_maps_expand.flatten(-2).to(device), dim=2)
        super_features, _ = scatter_max(src=input_features_in[:, :, 1:-1, 1:-1].flatten(-2).to(device),
                                       index=sp_maps_expand[:, :, 1:-1, 1:-1].flatten(-2).to(device).long(), dim=2)
        return super_features

id_x = 0

"""
Super attention block
"""
class Superattention_conv(nn.Module):
    def __init__(self, dims, in_channels):
        super(Superattention_conv, self).__init__()
        self.attent_soft = nn.Softmax(dim=dims)
        self.query_conv = nn.Conv1d( in_channels=in_channels,
                        out_channels=in_channels, kernel_size=1 )
        self.key_conv   = nn.Conv1d( in_channels=in_channels,
                        out_channels=in_channels, kernel_size=1 )
        self.value_conv = nn.Conv1d( in_channels=in_channels,
                        out_channels=in_channels, kernel_size=1)

    def forward(self, cat_encoded_target, cat_encoded_ref, label_mask_target, label_mask_ref, input_features_in, max_segments, cat_encoded_rgb_ref, max_segments_ref ,device ):

        """
        :param cat_encoded_target: Target segmented features of  size [B ,C, N] where N is the number of target's superpixels.
        :param cat_encoded_ref: Reference segmented features of  size [B, C, M] where M is the number of references superpixels.
        :param label_mask_target: Superpixel's target label map [B, C, H, W].
        :param label_mask_ref: Superpixel's reference label map [B, C, H, W].
        :param input_features_in: Features from previous convolutional block [B, C, H, W] and, it's used for tracking shape for the unpooling features.
        :param max_segments: Maximun number of superpixels.
        :return: wg_ref_feat: Weighted reference features of size [B, N, C] Where N is the number of target's segments.
        """

        batch_alpha = torch.zeros(len(cat_encoded_target), max_segments + 1, 1).to(device)
        attention_batch = torch.zeros(len(cat_encoded_target), max_segments + 1, max_segments_ref + 1).to(device)
        wg_ref_batch = torch.zeros(len(cat_encoded_target),  max_segments + 1, len(cat_encoded_target[0])).to(device)

        for batch_attention in range(len(cat_encoded_target)): # loop over batches
            eps = 1e-5

            # Normalizing input segmented features

            label_target = int(torch.max(label_mask_target[batch_attention, :, :])) + 1 # Retrieving current superpixels from the target N
            label_ref = int(torch.max(label_mask_ref[batch_attention, :, :])) + 1 # Retrieving current superpixels from the reference M

            encd_target = cat_encoded_target[batch_attention, :, 0:label_target].unsqueeze(0).to(device) # [B, C, N_max = M_max] ----> [B, C, N]
            encd_ref = cat_encoded_ref[batch_attention, :, 0:label_ref].unsqueeze(0).to(device) # [B, C, N_max = M_max] ----> [B, C, M]
            encd_rgb_ref = cat_encoded_rgb_ref[batch_attention, :, 0:label_ref].unsqueeze(0).to(device)

            # Feature's normalization
            encd_ref_mean = encd_ref - torch.mean(encd_ref, axis=1)
            encd_target_mean = encd_target - torch.mean(encd_target, axis=1)
            encd_target_norm = encd_target_mean / (torch.norm(encd_target_mean, p=2, keepdim=True,
                                                             dim=1) + eps)  # Normalize encoded features from target
            encd_ref_norm = encd_ref_mean / (torch.norm(encd_ref_mean, p=2, keepdim=True,
                                                       dim=1) + eps) # Normalize encoded features from reference

            encd_target_norm = self.query_conv(encd_target_norm)

            encd_ref_norm = self.key_conv(encd_ref_norm)


            # Traget transpose.. target [1, C , N_segments] ----> [1, N_segments, C]
            transpose_target = torch.transpose(encd_target_norm, 1, 2)


            # Similarity calculation target: [1, N_segments, C] --- Reference: [1, C, M_segments]
            sim = torch.bmm(transpose_target, encd_ref_norm)  # Sim: [B, N_segments, M_segments]

            sim_soft = self.attent_soft(sim/ 0.01)  # Attention map


            encd_rgb_ref = self.value_conv(encd_rgb_ref)
            transpose_ref = torch.transpose(encd_rgb_ref, 1, 2)

            # Weightening reference enconded features [B, N, C]
            wg_ref_feat = torch.bmm(sim_soft,
                                    transpose_ref)  # Result: sim_soft: [1, N_segments, M_segments] and [1, M_segments, C] = [1, N_segments, C]

            h_sim, w_sim = sim_soft.shape[1:]
            batch_alpha[batch_attention, 0:label_target, :] = 0
            attention_batch[batch_attention, 0:h_sim, 0:w_sim] = sim_soft[0, :, :]
            # attention_batch[batch_attention, 0:label_target, 0:label_ref] = sim_soft[0, :, :] # Stacking attention map [B, N_max, M_max]
            # wg_ref_batch[batch_attention, 0:label_target, :] = wg_ref_feat_alpha[0, :, :]  # Stacking Weighted reference into batches by filling up with zeros [B, N_max, C ]
            wg_ref_batch[batch_attention, 0:h_sim, :] = wg_ref_feat[0, :, :]
        wg_ref_batch_transp = torch.transpose(wg_ref_batch, 1, 2) #[B, C, N]
        return wg_ref_batch_transp, attention_batch, batch_alpha


class Unpool_features(nn.Module):

    def __init__(self):

        super(Unpool_features, self).__init__()

    def forward(self, cat_encoded_wg, shape_input_features_in, label_mask, device):
        """
        :param input_features: Feature tensor [B, C, H, W]
        :param cat_encoded_wg: Staked array's of masked tensor of size [B, C, N_max = M_max]
        :param label_mask: int label mask of size [B, C, H, W]
        :return cat_encoded: Staked array's of masked tensor of size [B, C, N_max = M_max]
        """

        batch_size, n_channels, H, W = shape_input_features_in
        sp_maps_expand = label_mask.expand(-1, n_channels, -1, -1).to(device)
        unpool_feat = torch.gather(cat_encoded_wg, 2, sp_maps_expand.flatten(-2))
        return torch.reshape(unpool_feat, (batch_size, n_channels, H, W))


"""
Generative model
"""

class gen_color_stride_vgg16(nn.Module):
    def __init__(self, dim):
        super(gen_color_stride_vgg16, self).__init__()

        norm_layer = nn.BatchNorm2d
        num_channel = [64, 128, 256, 512, 1024]
        num_channel_conv = [192, 384, 768, 1536]

        self.super_feat = Superfeatures()
        self.Unpool = Unpool_features()
        self.attention_1 = Superattention_conv(dims=dim, in_channels=num_channel[0])
        self.attention_2 = Superattention_conv(dims=dim, in_channels=num_channel[1])
        self.attention_3 = Superattention_conv(dims=dim, in_channels=num_channel[2])
        self.attention_4 = Superattention_conv(dims=dim, in_channels=num_channel[3])


        self.gen1 = nn.Sequential(nn.Conv2d(3, num_channel[0], 3, stride=1, padding=1),
                                  norm_layer(num_channel[0]),
                                  nn.ReLU(True),
                                  nn.Conv2d(num_channel[0], num_channel[0], 3, stride=1, padding=1),
                                  norm_layer(num_channel[0]),
                                  nn.ReLU(True))

        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.gen2 = nn.Sequential(nn.Conv2d(num_channel[0], num_channel[1], 3, stride=1, padding=1),
                                  norm_layer(num_channel[1]),
                                  nn.ReLU(True),
                                  nn.Conv2d(num_channel[1], num_channel[1], 3, stride=1, padding=1),
                                  norm_layer(num_channel[1]),
                                  nn.ReLU(True))



        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.gen3 = nn.Sequential(nn.Conv2d(num_channel[1], num_channel[2], 3, stride=1, padding=1),
                                  norm_layer(num_channel[2]),
                                  nn.ReLU(True),
                                  nn.Conv2d(num_channel[2], num_channel[2], 3, stride=1, padding=1),
                                  norm_layer(num_channel[2]),
                                  nn.ReLU(True),
                                  nn.Conv2d(num_channel[2], num_channel[2], 3, stride=1, padding=1),
                                  norm_layer(num_channel[2]),
                                  nn.ReLU(True))



        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.gen4 = nn.Sequential(nn.Conv2d(num_channel[2], num_channel[3], 3, stride=1, padding=1),
                                  norm_layer(num_channel[3]),
                                  nn.ReLU(True),
                                  nn.Conv2d(num_channel[3], num_channel[3], 3, stride=1, padding=1),
                                  norm_layer(num_channel[3]),
                                  nn.ReLU(True),
                                  nn.Conv2d(num_channel[3], num_channel[3], 3, stride=1, padding=1),
                                  norm_layer(num_channel[3]),
                                  nn.ReLU(True))



        self.gen4_up = nn.ConvTranspose2d(num_channel[3], num_channel[2], 4, stride=2, padding=1)

        self.gen5_conv = nn.Conv2d(3 * num_channel[2], num_channel[2], 1, stride=1, padding=0)

        self.gen5 = nn.Sequential(nn.Conv2d(num_channel[2], num_channel[2], 3, stride=1, padding=1),
                                  norm_layer(num_channel[2]),
                                  nn.ReLU(True),
                                  nn.Conv2d(num_channel[2], num_channel[2], 3, stride=1, padding=1),
                                  norm_layer(num_channel[2]),
                                  nn.ReLU(True),
                                  nn.Conv2d(num_channel[2], num_channel[2], 3, stride=1, padding=1),
                                  norm_layer(num_channel[2]),
                                  nn.ReLU(True))

        self.gen5_up = nn.Sequential(nn.ConvTranspose2d(num_channel[2], num_channel[1], 4, stride=2, padding=1),
                                     nn.ReLU(True))

        self.gen6_conv = nn.Conv2d(num_channel_conv[1], num_channel[1], 1, stride=1, padding=0)

        self.gen6 = nn.Sequential(nn.Conv2d(num_channel[1], num_channel[1], 3, stride=1, padding=1),
                                  norm_layer(num_channel[1]),
                                  nn.ReLU(True),
                                  nn.Conv2d(num_channel[1], num_channel[1], 3, stride=1, padding=1),
                                  norm_layer(num_channel[1]),
                                  nn.ReLU(True),
                                  nn.Conv2d(num_channel[1], num_channel[1], 3, stride=1, padding=1),
                                  norm_layer(num_channel[1]),
                                  nn.ReLU(True))

        self.gen6_up = nn.Sequential(nn.ConvTranspose2d(num_channel[1], num_channel[0], 4, stride=2, padding=1),
                                     nn.ReLU(True))

        self.gen7_conv = nn.Conv2d(num_channel_conv[0], num_channel[0], 1, stride=1, padding=0)

        self.gen7 = nn.Sequential(nn.Conv2d(num_channel[0], num_channel[0], 3, stride=1, padding=1),
                                  norm_layer(num_channel[0]),
                                  nn.ReLU(True),
                                  nn.Conv2d(num_channel[0], num_channel[0], 3, stride=1, padding=1),
                                  norm_layer(num_channel[0]),
                                  nn.ReLU(True),
                                  nn.Conv2d(num_channel[0], num_channel[0], 3, stride=1, padding=1),
                                  norm_layer(num_channel[0]),
                                  nn.ReLU(True))


        self.out = nn.Sequential(nn.Conv2d(num_channel[0], 2, 1, padding=0, dilation=1, stride=1),
                                 nn.Tanh())

    def forward(self, ref, luminance_target, label_mask_x, label_mask_ref, target_map, x_real, feat1_pred, feat2_pred, feat3_pred, device):



        label_max_1 = torch.max(label_mask_x[0])
        label_max_2 = torch.max(label_mask_x[1])
        label_max_3 = torch.max(label_mask_x[2])

        label_mask_ref_1 = torch.max(label_mask_ref[0])
        label_mask_ref_2 = torch.max(label_mask_ref[1])
        label_mask_ref_3 = torch.max(label_mask_ref[2])

        # ENCODER
        # Target image path
        block1 = self.gen1(target_map)
        block1_down = self.down1(block1)
        block2 = self.gen2(block1_down)
        block2_down = self.down2(block2)
        block3 = self.gen3(block2_down)
        block3_down = self.down3(block3)
        block4 = self.gen4(block3_down)


        # Reference image path
        block1_ref = self.gen1(ref)
        block1_down_ref = self.down1(block1_ref)
        block2_ref = self.gen2(block1_down_ref)
        block2_down_ref = self.down2(block2_ref)
        block3_ref = self.gen3(block2_down_ref)




        # Reference RGB image
        block1_rgb_ref = feat1_pred
        block2_rgb_ref = feat2_pred
        block3_rgb_ref = feat3_pred



        # SKIP CONNECTIONS
        encd_feat1_t = self.super_feat(block1, label_mask_x[0], device)
        encd_feat1_R = self.super_feat(block1_ref, label_mask_ref[0], device)
        encd_feat1_rgb_R = self.super_feat(block1_rgb_ref, label_mask_ref[0], device)
        superattention_1, sim_matrix_1, alpha_1 = self.attention_1(encd_feat1_t, encd_feat1_R, label_mask_x[0], label_mask_ref[0], block1, label_max_1, encd_feat1_rgb_R,label_mask_ref_1 ,device)
        wg_1 = self.Unpool(superattention_1, block1.shape, label_mask_x[0], device)

        encd_feat2_t = self.super_feat(block2, label_mask_x[1], device)
        encd_feat2_R = self.super_feat(block2_ref, label_mask_ref[1], device)
        encd_feat2_rgb_R = self.super_feat(block2_rgb_ref, label_mask_ref[1], device)
        superattention_2, sim_matrix_2, alpha_2 = self.attention_2(encd_feat2_t, encd_feat2_R, label_mask_x[1], label_mask_ref[1], block2, label_max_2, encd_feat2_rgb_R,label_mask_ref_2 ,device)
        wg_2 = self.Unpool(superattention_2, block2.shape, label_mask_x[1], device)

        encd_feat3_t = self.super_feat(block3, label_mask_x[2], device)
        encd_feat3_R = self.super_feat(block3_ref, label_mask_ref[2], device)
        encd_feat3_rgb_R = self.super_feat(block3_rgb_ref, label_mask_ref[2], device)
        superattention_3, sim_matrix_3, alpha_3 = self.attention_3(encd_feat3_t, encd_feat3_R, label_mask_x[2], label_mask_ref[2], block3, label_max_3, encd_feat3_rgb_R,label_mask_ref_3 ,device)
        wg_3 = self.Unpool(superattention_3, block3.shape, label_mask_x[2], device)


        # Decoder and skip connections
        block4_up = self.gen4_up(block4)  # Block 6
        block5_skip = torch.cat((block4_up, wg_3, block3),
                                dim=1)  # Concatenating over channels [B,256,H/4, W/4] ---> [B, 3*256, H/4, W/4]
        block5_conv = self.gen5_conv(block5_skip)  # [B,3*256,H/4, W/4] ---> [B, 256, H/4, W/4]
        block5 = self.gen5(block5_conv)  # [B,256,H/4, W/4]

        block5_up = self.gen5_up(block5)  # Block 7
        block6_skip = torch.cat((block5_up, wg_2, block2),
                                dim=1)  # Concatenating over channels [B,128,H/4, W/4] ---> [B, 3*128, H/4, W/4]
        block6_conv = self.gen6_conv(block6_skip)  # [B,3*128,H/4, W/4] ---> [B, 128, H/4, W/4]
        block6 = self.gen6(block6_conv)

        block6_up = self.gen6_up(block6)  # Block 8

        block7_skip = torch.cat((block6_up, wg_1, block1),
                                dim=1)  # Concatenating over channels [B,64,H/4, W/4] ---> [B, 3*64, H/4, W/4]

        block7_conv = self.gen7_conv(block7_skip)  # [B,3*64,H/4, W/4] ---> [B, 64, H/4, W/4]
        block7 = self.gen7(block7_conv)
        out_reg = self.out(block7)

        c, h, w = x_real[:, 0, :, :].shape
        out_unorm = (out_reg * 127.0)
        predictedAB = resize(out_unorm[0, :, :, :].cpu().detach().numpy(), (2, h, w))
        luminance_unnorm = (x_real[:, 0, :, :]*100.0)
        predictedAB = torch.tensor(predictedAB).to(device)
        pred_Lab_torch = torch.cat((luminance_target[:, 0, :, :].unsqueeze(1), out_reg), dim=1)
        pred_unorm_Lab_torch = torch.cat((luminance_unnorm.unsqueeze(1), predictedAB.unsqueeze(0)), dim=1)
        pred_RGB = kornia.color.lab_to_rgb(pred_unorm_Lab_torch, clip=True)


        return out_reg, pred_Lab_torch, pred_RGB