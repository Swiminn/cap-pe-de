import abc
import torch
import torch.nn as nn
import torchvision
from . import network_util as net_util


def get_backbone_dict():
    return {
        'res34fpn': ResNet34FPN,
        'res50fpn': ResNet50FPN,
        'res101fpn': ResNet101FPN,
        'vit_b_16': ViTB16
    }


class BackboneABC(abc.ABC, nn.Module):
    def __init__(self, network_args):
        super(BackboneABC, self).__init__()
        self.pretrained = network_args['pretrained']
        self.net = nn.ModuleDict()

    @ abc.abstractmethod
    def build(self):
        pass

    @ abc.abstractmethod
    def get_fmap2img_ratios(self):
        pass


class ResNet34FPN(BackboneABC):
    def __init__(self, network_args):
        super(ResNet34FPN, self).__init__(network_args)
        self.inter_chs = (128, 256, 512)
        self.fmap_ch = network_args['fmap_ch']

    def __get_base_network__(self):
        return torchvision.models.resnet34(pretrained=self.pretrained)

    def get_fmap2img_ratios(self):
        return 1/8.0, 1/16.0, 1/32.0, 1/64.0, 1/128.0

    def build(self):
        base_net = self.__get_base_network__()
        self.net['base'] = nn.Sequential(
            base_net.conv1,
            # nn.Conv2d(20, 64, 3, 2, 1, bias=False),
            base_net.bn1, base_net.relu,
            base_net.maxpool, base_net.layer1)

        self.net['stage_c3'] = base_net.layer2
        self.net['stage_c4'] = base_net.layer3
        self.net['stage_c5'] = base_net.layer4
        self.net['stage_c6'] = nn.Conv2d(self.inter_chs[2], self.fmap_ch, 3, 2, 1)
        self.net['stage_c7'] = nn.Sequential(
            nn.ReLU(), nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 2, 1))

        self.net['stage_p5_1'] = nn.Conv2d(self.inter_chs[2], self.fmap_ch, 1, 1, 0)
        self.net['stage_p5_2'] = nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 1, 1)
        self.net['stage_p5_up'] = nn.Upsample(scale_factor=2, mode='nearest')
        # self.net['stage_p5_up'] = nn.ConvTranspose2d(self.fmap_ch, self.fmap_ch, 3, 2, 1, 1)

        self.net['stage_p4_1'] = nn.Conv2d(self.inter_chs[1], self.fmap_ch, 1, 1, 0)
        self.net['stage_p4_2'] = nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 1, 1)
        self.net['stage_p4_up'] = nn.Upsample(scale_factor=2, mode='nearest')
        # self.net['stage_p4_up'] = nn.ConvTranspose2d(self.fmap_ch, self.fmap_ch, 3, 2, 1, 1)

        self.net['stage_p3_1'] = nn.Conv2d(self.inter_chs[0], self.fmap_ch, 1, 1, 0)
        self.net['stage_p3_2'] = nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 1, 1)

        if self.pretrained:
            print('[BACKBONE] load image-net pre-trained model')
        else:
            net_util.init_modules_xavier(
                [self.net['base'], self.net['stage_c3'],
                 self.net['stage_c4'], self.net['stage_c5']])
        net_util.init_modules_xavier(
            [self.net['stage_c6'], self.net['stage_c7'],
             self.net['stage_p5_1'], self.net['stage_p5_2'],
             self.net['stage_p4_1'], self.net['stage_p4_2'],
             self.net['stage_p3_1'], self.net['stage_p3_2']])

    def forward(self, image, num_level=5):
        base_fmap = self.net['base'].forward(image)
        fmap_c3 = self.net['stage_c3'].forward(base_fmap)
        fmap_c4 = self.net['stage_c4'].forward(fmap_c3)
        fmap_c5 = self.net['stage_c5'].forward(fmap_c4)
        fmap_p6 = self.net['stage_c6'].forward(fmap_c5)
        fmap_p7 = self.net['stage_c7'].forward(fmap_p6)

        _fmap_p5 = self.net['stage_p5_1'].forward(fmap_c5)
        fmap_p5 = self.net['stage_p5_2'].forward(_fmap_p5)
        _fmap_p5_up = self.net['stage_p5_up'].forward(_fmap_p5)

        _fmap_p4 = self.net['stage_p4_1'].forward(fmap_c4) + _fmap_p5_up
        fmap_p4 = self.net['stage_p4_2'].forward(_fmap_p4)
        _fmap_p4_up = self.net['stage_p4_up'].forward(_fmap_p4)

        _fmap_p3 = self.net['stage_p3_1'].forward(fmap_c3) + _fmap_p4_up
        fmap_p3 = self.net['stage_p3_2'].forward(_fmap_p3)
        return [fmap_p3, fmap_p4, fmap_p5, fmap_p6, fmap_p7][:num_level]


class ResNet50FPN(ResNet34FPN):
    def __init__(self, network_args):
        super(ResNet50FPN, self).__init__(network_args)
        self.inter_chs = (512, 1024, 2048)

    def __get_base_network__(self):
        return torchvision.models.resnet50(pretrained=self.pretrained)


class ResNet101FPN(ResNet50FPN):
    def __get_base_network__(self):
        return torchvision.models.resnet101(pretrained=self.pretrained)


class ViTB16(BackboneABC):
    def __init__(self, network_args):
        super(ViTB16, self).__init__(network_args)
        # self.inter_chs = (128, 256, 512)
        self.fmap_ch = network_args['fmap_ch']
        self.hidden_dim = network_args['hidden_dim']
        self.patch_size = network_args['patch_size']
        self.img_h = 800
        self.img_w = 800

    def __get_base_network__(self):
        return torchvision.models.vit_b_16(pretrained=self.pretrained)

    def get_fmap2img_ratios(self):
        return 1/8.0, 1/16.0, 1/32.0, 1/64.0, 1/128.0

    def build(self):
        base_net = self.__get_base_network__()
        self.net['vit'] = base_net
        self.net['conv_proj'] = base_net.conv_proj
        self.net['encoder_layers'] = base_net.encoder.layers
        self.net['layer_norm'] = base_net.encoder.ln
        # self.net['heads'] = base_net.heads

        # TODO
        # encoder output shape: [16, 2500, 786]
        # head output shape: [16, 2500, 1000]

        self.net['stage_out_1'] = nn.Conv2d(self.hidden_dim, self.fmap_ch, 1, 1, 0)
        self.net['stage_out_pool'] = nn.MaxPool2d(8, 8, ceil_mode=True)

        self.net['stage_p10_1'] = nn.Conv2d(self.hidden_dim, self.fmap_ch, 1, 1, 0)
        self.net['stage_p10_pool'] = nn.MaxPool2d(4, 4, ceil_mode=True)

        self.net['stage_p9_1'] = nn.Conv2d(self.hidden_dim, self.fmap_ch, 1, 1, 0)
        self.net['stage_p9_2'] = nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 1, 1)
        self.net['stage_p9_up'] = nn.Upsample(scale_factor=1, mode='nearest')
        self.net['stage_p9_pool'] = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.net['stage_p8_1'] = nn.Conv2d(self.hidden_dim, self.fmap_ch, 1, 1, 0)
        self.net['stage_p8_2'] = nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 1, 1)
        self.net['stage_p8_up'] = nn.Upsample(scale_factor=1, mode='nearest')

        self.net['stage_p7_1'] = nn.Conv2d(self.hidden_dim, self.fmap_ch, 1, 1, 0)
        self.net['stage_p7_2'] = nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 1, 1)
        self.net['stage_p7_up'] = nn.Upsample(scale_factor=2, mode='nearest')

    # https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        self.img_h = h
        self.img_w = w
        p = self.patch_size
        # torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        # torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        # print(f"x shape previous: {x.shape}")
        x = self.net['conv_proj'](x)
        # print(f"x shape new: {x.shape}")
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    # TODO
    def _process_output(self, x: torch.Tensor) -> torch.Tensor:
        n, _, _ = x.shape
        p = self.patch_size
        # torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        # torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = self.img_h // p
        n_w = self.img_w // p

        # print(f"x shape previous: {x.shape}")

        x = x.permute(0, 2, 1)
        x = x.reshape(n, self.hidden_dim, n_h, n_w)

        # print(f"x shape new: {x.shape}")

        return x

    def forward(self, image, num_level=5):

        assert num_level > 0, "'num_level' must be larger than zero"

        # print(f"Image shape: {image.shape}")
        base_fmap = self._process_input(image)
        # print(f"base_fmap: {base_fmap.shape}")

        encoder_0 = self.net['encoder_layers'][0].forward(base_fmap)
        encoder_1 = self.net['encoder_layers'][1].forward(encoder_0)
        encoder_2 = self.net['encoder_layers'][2].forward(encoder_1)
        encoder_3 = self.net['encoder_layers'][3].forward(encoder_2)
        encoder_4 = self.net['encoder_layers'][4].forward(encoder_3)
        encoder_5 = self.net['encoder_layers'][5].forward(encoder_4)
        encoder_6 = self.net['encoder_layers'][6].forward(encoder_5)
        encoder_7 = self.net['encoder_layers'][7].forward(encoder_6)
        encoder_8 = self.net['encoder_layers'][8].forward(encoder_7)
        encoder_9 = self.net['encoder_layers'][9].forward(encoder_8)
        encoder_10 = self.net['encoder_layers'][10].forward(encoder_9)
        encoder_output = self.net['encoder_layers'][11].forward(encoder_10)
        _output = self.net['layer_norm'](encoder_output)
        # output = self.net['heads'].forward(encoder_output)

        encoder_7 = self._process_output(encoder_7)
        encoder_8 = self._process_output(encoder_8)
        encoder_9 = self._process_output(encoder_9)
        encoder_10 = self._process_output(encoder_10)
        _output = self._process_output(_output)

        _output = self.net['stage_out_1'].forward(_output)
        encoder_10 = self.net['stage_p10_1'].forward(encoder_10)

        _encoder_p9 = self.net['stage_p9_1'].forward(encoder_9)
        encoder_p9 = self.net['stage_p9_2'].forward(_encoder_p9)
        # _encoder_p9_up = self.net['stage_p9_up'].forward(_encoder_p9)

        _encoder_p8 = self.net['stage_p8_1'].forward(encoder_8) + _encoder_p9
        encoder_p8 = self.net['stage_p8_2'].forward(_encoder_p8)
        # _encoder_p8_up = self.net['stage_p8_up'].forward(_encoder_p8)

        # [batch_size, fmap_ch, img_h//p * 2, img_w//p * 2]
        _encoder_p7 = self.net['stage_p7_1'].forward(encoder_7) + _encoder_p8
        encoder_p7 = self.net['stage_p7_2'].forward(_encoder_p7)
        encoder_p7 = self.net['stage_p7_up'].forward(encoder_p7)

        # Downsizing
        encoder_p9 = self.net['stage_p9_pool'].forward(encoder_p9)
        encoder_10 = self.net['stage_p10_pool'].forward(encoder_10)
        _output = self.net['stage_out_pool'].forward(_output)

        output_zoo = [encoder_p7, encoder_p8, encoder_p9, encoder_10, _output][:num_level]

        return output_zoo


class ViTB32(ViTB16):
    def __get_base_network__(self):
        return torchvision.models.vit_b_32(pretrained=self.pretrained)
