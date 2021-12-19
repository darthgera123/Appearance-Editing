import math, cv2
import numpy as np
import sys
import torch
import torch.nn as nn

import torch.nn.functional as F

sys.path.append('..')
from model.texture import Texture, TextureMapper
from model.unet import TestUNet,UNet
from model.sh import *

class PipeLine(nn.Module):
    def __init__(self, W, H, feature_num, use_pyramid=True, view_direction=True, samples=10):
        super(PipeLine, self).__init__()
        self.feature_num = feature_num
        self.use_pyramid = use_pyramid
        self.view_direction = view_direction
        self.samples = samples
        self.texture = Texture(W, H, feature_num, use_pyramid)
        self.albedo_tex = Texture(W, H, 3, False)
        self.unet = TestUNet(feature_num, 9*3)

    def _spherical_harmonics_basis(self, extrinsics):
        '''
        extrinsics: a tensor shaped (N, 3)
        output: a tensor shaped (N, 9)
        '''
        batch = extrinsics.shape[0]
        sh_bands = torch.ones((batch, 9), dtype=torch.float)
        
        coff_0 = 1 / (2.0 * math.sqrt(np.pi))
        coff_1 = math.sqrt(3.0) * coff_0
        coff_2 = math.sqrt(15.0) * coff_0
        coff_3 = math.sqrt(1.25) * coff_0
        # l=0
        sh_bands[:, 0] = coff_0
        # l=1
        sh_bands[:, 1] = extrinsics[:, 1] * coff_1 * math.sqrt(0.5)
        sh_bands[:, 2] = extrinsics[:, 2] * coff_1
        sh_bands[:, 3] = extrinsics[:, 0] * coff_1 * (-1) * math.sqrt(0.5)
        # l=2
        sh_bands[:, 4] = extrinsics[:, 0] * extrinsics[:, 1] * coff_2 * 0.5 * math.sqrt(0.5)
        sh_bands[:, 5] = extrinsics[:, 1] * extrinsics[:, 2] * coff_2 * math.sqrt(0.5)
        sh_bands[:, 6] = (3.0 * extrinsics[:, 2] * extrinsics[:, 2] - 1.0) * coff_3
        sh_bands[:, 7] = extrinsics[:, 2] * extrinsics[:, 0] * coff_2 * (-1) * math.sqrt(0.5)
        sh_bands[:, 8] = (extrinsics[:, 0] * extrinsics[:, 0] - extrinsics[:, 2] * extrinsics[:,
                                                                                   2]) * coff_2 * 0.5 * math.sqrt(0.5)
        return sh_bands

    def forward(self, *args):
        wi, envmap, uv_map, extrinsics = args

       
        # Make Diffuse BRDF
        albedo = self.albedo_tex(uv_map)  # [b, 3, h, w]
        albedo_ = torch.unsqueeze(albedo, dim=2)  # [b, 3, 1, h, w]
        albedo_ = torch.tile(albedo_, (1, 1, wi.shape[2], 1, 1))  # [b, 3, samples, h, w]
        # albedo_ = albedo_ * cos_t

        # Forward
        forward = albedo_ * envmap
        forward = torch.mean(forward, dim=2, keepdim=False) # [b, 3, h, w]

        # SH project
        y_0_0 = torch.mean( y_0_0_(albedo_, wi[:, 0:1, :, :, :], wi[:, 1:2, :, :, :]), dim=2 ) # [b, 3, h, w]

        y_1_n1 = torch.mean( y_1_n1_(albedo_, wi[:, 0:1, :, :, :], wi[:, 1:2, :, :, :]), dim=2 )
        y_1_0 = torch.mean( y_1_0_(albedo_, wi[:, 0:1, :, :, :], wi[:, 1:2, :, :, :]), dim=2 )
        y_1_p1 = torch.mean( y_1_p1_(albedo_, wi[:, 0:1, :, :, :], wi[:, 1:2, :, :, :]), dim=2 )

        y_2_n2 = torch.mean( y_2_n2_(albedo_, wi[:, 0:1, :, :, :], wi[:, 1:2, :, :, :]), dim=2 )
        y_2_n1 = torch.mean( y_2_n1_(albedo_, wi[:, 0:1, :, :, :], wi[:, 1:2, :, :, :]), dim=2 )
        y_2_0 = torch.mean( y_2_0_(albedo_, wi[:, 0:1, :, :, :], wi[:, 1:2, :, :, :]), dim=2 )
        y_2_p1 = torch.mean( y_2_p1_(albedo_, wi[:, 0:1, :, :, :], wi[:, 1:2, :, :, :]), dim=2 )
        y_2_p2 = torch.mean( y_2_p2_(albedo_, wi[:, 0:1, :, :, :], wi[:, 1:2, :, :, :]), dim=2 )

        f = torch.cat((y_0_0, y_1_n1, y_1_0, y_1_p1, y_2_n2, y_2_n1, y_2_0, y_2_p1, y_2_p2), dim=1) # [b, 3*9, h, w]

        # Neural texture sampling
        nt = self.texture(uv_map)
        basis = self._spherical_harmonics_basis(extrinsics).cuda()
        basis = basis.view(basis.shape[0], basis.shape[1], 1, 1)
        nt[:, 3:12, :, :] = nt[:, 3:12, :, :] * basis

        # NN forward
        T = self.unet(nt)  # [b, 3*samples, h, w]  # Transport / Transfer

        final = f.view(-1, 9, 3, f.shape[2], f.shape[3]) * T.view(-1, 9, 3, T.shape[2], T.shape[3]) # [b, 9, 3, h, w]
        final = torch.sum(final, dim=1, keepdim=False) # [b, 3, h, w]

        albedo_tex = torch.cat((self.albedo_tex.textures[0].layer1, self.albedo_tex.textures[1].layer1, self.albedo_tex.textures[2].layer1), dim=1)
        return albedo_tex, final, forward, albedo 



class PipeLineTex(nn.Module):
    def __init__(self, W, H, feature_num, use_pyramid=True, view_direction=True, samples=10):
        super(PipeLine, self).__init__()
        self.feature_num = feature_num
        self.use_pyramid = use_pyramid
        self.view_direction = view_direction
        self.samples = samples

        self.albedo_tex = Texture(W*2, H*2, 3, False)

    def forward(self, *args):
        wi, cos_t, envmap, uv_map = args

        # wi : [b, 2, samples, h, w]
        # cos_t : [b, samples, h, w]
        # envmap : [b, 3, samples, h, w]

        wi = wi.view(-1, 2 * self.samples, wi.shape[3], wi.shape[4])  # [b, 2*samples, h, w]
        cos_t = torch.unsqueeze(cos_t, dim=1)  # [b, 1, samples, h, w]
        cos_t = torch.tile(cos_t, (1, 3, 1, 1, 1))  # [b, 3, samples, h, w]

        # Make Diffuse BRDF
        albedo = self.albedo_tex(uv_map)  # [b, 3, h, w]
        albedo_ = torch.unsqueeze(albedo, dim=2)  # [b, 3, 1, h, w]
        albedo_ = torch.tile(albedo_, (1, 1, cos_t.shape[2], 1, 1))  # [b, 3, samples, h, w]

        final = albedo_ * cos_t * envmap
        final = 2.0 / float(self.samples) * torch.sum(final, dim=2)  # 10 - samples

        albedo_tex = torch.cat((self.albedo_tex.textures[0].layer1, self.albedo_tex.textures[1].layer1, self.albedo_tex.textures[2].layer1), dim=1)

        return albedo_tex, albedo, final, cos_t


class PipeLineMask(nn.Module):
    def __init__(self, W, H, feature_num, use_pyramid=True, view_direction=True):
        super(PipeLineMask, self).__init__()
        self.feature_num = feature_num
        self.use_pyramid = use_pyramid
        self.view_direction = view_direction
        self.texture = Texture(W, H, feature_num, use_pyramid)
        self.unet = UNet(feature_num, 1)

    def _spherical_harmonics_basis(self, extrinsics):
        '''
        extrinsics: a tensor shaped (N, 3)
        output: a tensor shaped (N, 9)
        '''
        batch = extrinsics.shape[0]
        sh_bands = torch.ones((batch, 9), dtype=torch.float)
        coff_0 = 1 / (2.0*math.sqrt(np.pi))
        coff_1 = math.sqrt(3.0) * coff_0
        coff_2 = math.sqrt(15.0) * coff_0
        coff_3 = math.sqrt(1.25) * coff_0
        # l=0
        sh_bands[:, 0] = coff_0
        # l=1
        sh_bands[:, 1] = extrinsics[:, 1] * coff_1
        sh_bands[:, 2] = extrinsics[:, 2] * coff_1
        sh_bands[:, 3] = extrinsics[:, 0] * coff_1
        # l=2
        sh_bands[:, 4] = extrinsics[:, 0] * extrinsics[:, 1] * coff_2
        sh_bands[:, 5] = extrinsics[:, 1] * extrinsics[:, 2] * coff_2
        sh_bands[:, 6] = (3.0 * extrinsics[:, 2] * extrinsics[:, 2] - 1.0) * coff_3
        sh_bands[:, 7] = extrinsics[:, 2] * extrinsics[:, 0] * coff_2
        sh_bands[:, 8] = (extrinsics[:, 0] * extrinsics[:, 0] - extrinsics[:, 2] * extrinsics[:, 2]) * coff_2
        return sh_bands

    def forward(self, *args):
        if self.view_direction:
            uv_map, extrinsics = args
            x = self.texture(uv_map)
            assert x.shape[1] >= 12
            basis = self._spherical_harmonics_basis(extrinsics).cuda()
            basis = basis.view(basis.shape[0], basis.shape[1], 1, 1)
            x[:, 3:12, :, :] = x[:, 3:12, :, :] * basis
        else:
            uv_map = args[0]
            x = self.texture(uv_map)
        y = self.unet(x)
        return x[:, 0:3, :, :], y