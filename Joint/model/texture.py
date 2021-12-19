import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('..')

class SingleLayerTexture(nn.Module):
    def __init__(self, W, H):
        super(SingleLayerTexture, self).__init__()
        self.layer1 = nn.Parameter(torch.zeros((1, 1, W, H), dtype=torch.float))

    def forward(self, x):
        batch = x.shape[0]
        x = x * 2.0 - 1.0
        y = F.grid_sample(self.layer1.repeat(batch,1,1,1), x, align_corners=True)
        return y


class LaplacianPyramid(nn.Module):
    def __init__(self, W, H):
        super(LaplacianPyramid, self).__init__()
        self.layer1 = nn.Parameter(torch.zeros((1, 1, W, H), dtype=torch.float))
        self.layer2 = nn.Parameter(torch.zeros((1, 1, W // 2, H // 2), dtype=torch.float))
        self.layer3 = nn.Parameter(torch.zeros((1, 1, W // 4, H // 4), dtype=torch.float))
        self.layer4 = nn.Parameter(torch.zeros((1, 1, W // 8, H // 8), dtype=torch.float))

    def forward(self, x):
        batch = x.shape[0]
        x = x * 2.0 - 1.0
        y1 = F.grid_sample(self.layer1.repeat(batch,1,1,1), x, align_corners=True)
        y2 = F.grid_sample(self.layer2.repeat(batch,1,1,1), x, align_corners=True)
        y3 = F.grid_sample(self.layer3.repeat(batch,1,1,1), x, align_corners=True)
        y4 = F.grid_sample(self.layer4.repeat(batch,1,1,1), x, align_corners=True)
        y = y1 + y2 + y3 + y4
        return y


class Texture(nn.Module):
    def __init__(self, W, H, feature_num, use_pyramid=True):
        super(Texture, self).__init__()
        self.feature_num = feature_num
        self.use_pyramid = use_pyramid
        self.layer1 = nn.ParameterList()
        self.layer2 = nn.ParameterList()
        self.layer3 = nn.ParameterList()
        self.layer4 = nn.ParameterList()
        if self.use_pyramid:
            self.textures = nn.ModuleList([LaplacianPyramid(W, H) for i in range(feature_num)])
            for i in range(self.feature_num):
                self.layer1.append(self.textures[i].layer1)
                self.layer2.append(self.textures[i].layer2)
                self.layer3.append(self.textures[i].layer3)
                self.layer4.append(self.textures[i].layer4)
        else:
            self.textures = nn.ModuleList([SingleLayerTexture(W, H) for i in range(feature_num)])
            for i in range(self.feature_num):
                self.layer1.append(self.textures[i].layer1)
        
    def forward(self, x):
        y_i = []
        for i in range(self.feature_num):
            y = self.textures[i](x)
            y_i.append(y)
        y = torch.cat(tuple(y_i), dim=1)
        return y

def interpolate_bilinear(data, sub_x, sub_y):
    '''
    data: [H, W, C]
    sub_x: [...]
    sub_y: [...]
    return: [..., C]
    '''
    device = data.device

    mask_valid = ((sub_x >= 0) & (sub_x <= data.shape[1] - 1) & (sub_y >= 0) & (sub_y <= data.shape[0] - 1)).to(data.dtype).to(device)

    x0 = torch.floor(sub_x).long().to(device)
    x1 = x0 + 1
    
    y0 = torch.floor(sub_y).long().to(device)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, data.shape[1] - 1)
    x1 = torch.clamp(x1, 0, data.shape[1] - 1)
    y0 = torch.clamp(y0, 0, data.shape[0] - 1)
    y1 = torch.clamp(y1, 0, data.shape[0] - 1)
    
    I00 = data[y0, x0, :] # [..., C]
    I10 = data[y1, x0, :]
    I01 = data[y0, x1, :]
    I11 = data[y1, x1, :]

    # right boundary
    x0 = x0 - (x0 == x1).to(x0.dtype)
    # bottom boundary
    y0 = y0 - (y0 == y1).to(y0.dtype)

    w00 = (x1.to(data.dtype) - sub_x) * (y1.to(data.dtype) - sub_y) * mask_valid # [...]
    w10 = (x1.to(data.dtype) - sub_x) * (sub_y - y0.to(data.dtype)) * mask_valid
    w01 = (sub_x - x0.to(data.dtype)) * (y1.to(data.dtype) - sub_y) * mask_valid
    w11 = (sub_x - x0.to(data.dtype)) * (sub_y - y0.to(data.dtype)) * mask_valid

    return I00 * w00.unsqueeze_(-1) + I10 * w10.unsqueeze_(-1) + I01 * w01.unsqueeze_(-1) + I11 * w11.unsqueeze_(-1)

class TextureMapper(nn.Module):
    def __init__(self,
                texture_size,
                texture_num_ch,
                mipmap_level, 
                texture_init = None,
                fix_texture = False,
                apply_sh = False):
        '''
        texture_size: [1]
        texture_num_ch: [1]
        mipmap_level: [1]
        texture_init: torch.FloatTensor, [H, W, C]
        apply_sh: bool, [1]
        '''
        super(TextureMapper, self).__init__()

        self.register_buffer('texture_size', torch.tensor(texture_size))
        self.register_buffer('texture_num_ch', torch.tensor(texture_num_ch))
        self.register_buffer('mipmap_level', torch.tensor(mipmap_level))
        self.register_buffer('apply_sh', torch.tensor(apply_sh))

        # create textures as images
        self.textures = nn.ParameterList([])
        self.textures_size = []
        for ithLevel in range(self.mipmap_level):
            texture_size_i = np.round(self.texture_size.numpy() / (2.0 ** ithLevel)).astype(np.int)
            texture_i = torch.ones(1, texture_size_i, texture_size_i, self.texture_num_ch, dtype = torch.float32)
            if ithLevel != 0:
                texture_i = texture_i * 0.01
            # initialize texture
            if texture_init is not None and ithLevel == 0:
                print('Initialize neural texture with reconstructed texture')
                texture_i[..., :texture_init.shape[-1]] = texture_init[None, :]
                texture_i[..., texture_init.shape[-1]:texture_init.shape[-1] * 2] = texture_init[None, :]
            self.textures_size.append(texture_size_i)
            self.textures.append(nn.Parameter(texture_i))

        tex_flatten_mipmap_init = self.flatten_mipmap(start_ch = 0, end_ch = 6)
        tex_flatten_mipmap_init = torch.nn.functional.relu(tex_flatten_mipmap_init)
        self.register_buffer('tex_flatten_mipmap_init', tex_flatten_mipmap_init)

        if fix_texture:
            print('Fix neural textures.')
            for i in range(self.mipmap_level):
                self.textures[i].requires_grad = False

    def forward(self, uv_map, sh_basis_map = None, sh_start_ch = 3):
        '''
        uv_map: [N, H, W, C]
        sh_basis_map: [N, H, W, 9]
        return: [N, C, H, W]
        '''
        for ithLevel in range(self.mipmap_level):
            texture_size_i = self.textures_size[ithLevel]
            texture_i = self.textures[ithLevel]

            # vertex texcoords map in unit of texel
            uv_map_unit_texel = (uv_map * (texture_size_i - 1))
            uv_map_unit_texel[..., -1] = texture_size_i - 1 - uv_map_unit_texel[..., -1]

            # sample from texture (bilinear)
            if ithLevel == 0:
                output = interpolate_bilinear(texture_i[0, :], uv_map_unit_texel[..., 0], uv_map_unit_texel[..., 1]).permute((0, 3, 1, 2)) # [N, C, H, W]
            else:
                output = output + interpolate_bilinear(texture_i[0, :], uv_map_unit_texel[..., 0], uv_map_unit_texel[..., 1]).permute((0, 3, 1, 2)) # [N, C, H, W]

        # apply spherical harmonics
        if self.apply_sh and sh_basis_map is not None:
            output[:, sh_start_ch:sh_start_ch + 9, :, :] = output[:, sh_start_ch:sh_start_ch + 9, :, :] * sh_basis_map.permute((0, 3, 1, 2))

        return output

    def flatten_mipmap(self, start_ch, end_ch):
        for ithLevel in range(self.mipmap_level):
            if ithLevel == 0:
                out = self.textures[ithLevel][..., start_ch:end_ch]
            else:
                out = out + torch.nn.functional.interpolate(self.textures[ithLevel][..., start_ch:end_ch].permute(0, 3, 1, 2), size = (self.textures_size[0], self.textures_size[0]), mode = 'bilinear').permute(0, 2, 3, 1)
        return out