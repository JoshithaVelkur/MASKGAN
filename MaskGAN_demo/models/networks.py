import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from torchvision import models


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[], device=None):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                             n_local_enhancers, n_blocks_local, norm_layer)
    else:
        raise NotImplementedError(f"Generator network [{netG}] not recognized")
    
    netG.apply(weights_init)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() and len(gpu_ids) > 0 else "cpu")
    netG.to(device)

    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1,
             getIntermFeat=False, gpu_ids=[], device=None):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    netD.apply(weights_init)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() and len(gpu_ids) > 0 else "cpu")
    netD.to(device)
    
    return netD

def define_B(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=3, norm='instance',
             gpu_ids=[], device=None):
    norm_layer = get_norm_layer(norm_type=norm)
    netB = BlendGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    netB.apply(weights_init)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() and len(gpu_ids) > 0 else "cpu")
    netB.to(device)

    return netB

def define_VAE(input_nc, gpu_ids=[], device=None):
    netVAE = VAE(19, 32, 32, 1024)  # Hardcoded settings
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() and len(gpu_ids) > 0 else "cpu")
    netVAE.to(device)
    return netVAE

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, device=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label).to(self.device)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label).to(self.device)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self, device=None):
        super(VGGLoss, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = Vgg19().to(self.device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

##############################################################################
# Generator
##############################################################################
class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        super(GlobalGenerator, self).__init__()
        assert n_blocks >= 0
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2), activation
            ]
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_type='adain', padding_type=padding_type)]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(int(ngf * mult / 2)), activation
            ]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

        self.enc_style = StyleEncoder(5, 3, 16, self.get_num_adain_params(self.model),
                                      norm='none', activ='relu', pad_type='reflect')
        self.enc_label = LabelEncoder(5, 19, 16, 64,
                                      norm='none', activ='relu', pad_type='reflect')

    def assign_adain_params(self, adain_params, model):
        for m in model.modules():
            if isinstance(m, AdaptiveInstanceNorm2d):
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        num_adain_params = 0
        for m in model.modules():
            if isinstance(m, AdaptiveInstanceNorm2d):
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def forward(self, input, input_ref, image_ref):
        fea1, fea2 = self.enc_label(input_ref)
        adain_params = self.enc_style((image_ref, fea1, fea2))
        self.assign_adain_params(adain_params, self.model)
        return self.model(input)


class BlendGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=3,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(BlendGenerator, self).__init__()
        assert n_blocks >= 0
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2), activation
            ]
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_type='in', padding_type=padding_type)]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(int(ngf * mult / 2)), activation
            ]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, input1, input2):
        m = self.model(torch.cat([input1, input2], dim=1))
        return input1 * m + input2 * (1 - m), m            

# Define the Multiscale Discriminator.
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, f'scale{i}_layer{j}', getattr(netD, f'model{j}'))
            else:
                setattr(self, f'layer{i}', netD.model)

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for layer in model:
                result.append(layer(result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        result = []
        input_downsampled = input
        for i in range(self.num_D):
            if self.getIntermFeat:
                model = [getattr(self, f'scale{self.num_D - 1 - i}_layer{j}') for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, f'layer{self.num_D - 1 - i}')
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (self.num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf), nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, f'model{n}', nn.Sequential(*sequence[n]))
        else:
            stream = []
            for s in sequence:
                stream += s
            self.model = nn.Sequential(*stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, f'model{n}')
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


# Define the MaskVAE
class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # Encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.e2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.e3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.e4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.e5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf * 16)
        self.e6 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1)
        self.bn6 = nn.BatchNorm2d(ndf * 32)
        self.e7 = nn.Conv2d(ndf * 32, ndf * 64, 4, 2, 1)
        self.bn7 = nn.BatchNorm2d(ndf * 64)

        self.fc1 = nn.Linear(ndf * 64 * 4 * 4, latent_variable_size)
        self.fc2 = nn.Linear(ndf * 64 * 4 * 4, latent_variable_size)

        # Decoder
        self.d1 = nn.Linear(latent_variable_size, ngf * 64 * 4 * 4)
        self.up = lambda: nn.Upsample(scale_factor=2, mode='nearest')
        self.pad = lambda: nn.ReplicationPad2d(1)
        self.decode_block = lambda in_ch, out_ch: nn.Sequential(
            self.pad(), nn.Conv2d(in_ch, out_ch, 3, 1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2)
        )

        self.dec_layers = nn.Sequential(
            self.decode_block(ngf * 64, ngf * 32),
            self.up(),
            self.decode_block(ngf * 32, ngf * 16),
            self.up(),
            self.decode_block(ngf * 16, ngf * 8),
            self.up(),
            self.decode_block(ngf * 8, ngf * 4),
            self.up(),
            self.decode_block(ngf * 4, ngf * 2),
            self.up(),
            self.decode_block(ngf * 2, ngf),
            self.up(),
            self.pad(),
            nn.Conv2d(ngf, nc, 3, 1)
        )

    def encode(self, x):
        h = self.e7(self.bn7(F.leaky_relu(self.e6(self.bn6(F.leaky_relu(self.e5(
            self.bn5(F.leaky_relu(self.e4(self.bn4(F.leaky_relu(self.e3(
            self.bn3(F.leaky_relu(self.e2(self.bn2(F.leaky_relu(self.e1(x)))))))))))))))))))
        h = h.view(h.size(0), -1)
        return self.fc1(h), self.fc2(h)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = F.relu(self.d1(z)).view(z.size(0), self.ngf * 64, 4, 4)
        return self.dec_layers(h)
    
    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar.mul(0.5).exp_()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), x, mu, logvar
    
class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        layers = [ConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for _ in range(2):
            layers.append(ConvBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
            dim *= 2

        mid_layers = [ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
                      for _ in range(n_downsample - 2)]

        self.model = nn.Sequential(*layers)
        self.model_middle = nn.Sequential(*mid_layers)
        self.model_last = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, style_dim, 1, 1, 0)
        )

        self.output_dim = dim
        self.sft1 = SFTLayer()
        self.sft2 = SFTLayer()

    def forward(self, x):
        fea = self.model(x[0])
        fea = self.sft1((fea, x[1]))
        fea = self.model_middle(fea)
        fea = self.sft2((fea, x[2]))
        return self.model_last(fea)


class LabelEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(LabelEncoder, self).__init__()
        layers = [
            ConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type),
            ConvBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type),
        ]
        dim *= 2
        layers.append(ConvBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation='none', pad_type=pad_type))
        dim *= 2

        post_layers = [nn.ReLU()]
        for _ in range(n_downsample - 3):
            post_layers.append(ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
        post_layers.append(ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation='none', pad_type=pad_type))

        self.model = nn.Sequential(*layers)
        self.model_last = nn.Sequential(*post_layers)
        self.output_dim = dim

    def forward(self, x):
        fea = self.model(x)
        return fea, self.model_last(fea)

# Define the basic block
class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(ConvBlock, self).__init__()
        self.use_bias = True

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            raise NotImplementedError(f"Unsupported pad_type: {pad_type}")

        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm in ['none', 'sn']:
            self.norm = None
        else:
            raise NotImplementedError(f"Unsupported norm: {norm}")

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise NotImplementedError(f"Unsupported activation: {activation}")

        conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        self.conv = SpectralNorm(conv) if norm == 'sn' else conv

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    
class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.fc = SpectralNorm(fc) if norm == 'sn' else fc

        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm in ['none', 'sn']:
            self.norm = None
        else:
            raise NotImplementedError(f"Unsupported norm: {norm}")

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise NotImplementedError(f"Unsupported activation: {activation}")

    def forward(self, x):
        x = self.fc(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_type='in', padding_type='reflect', use_dropout=False):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(dim, dim, 3, 1, 1, norm=norm_type, activation='relu', pad_type=padding_type),
            ConvBlock(dim, dim, 3, 1, 1, norm=norm_type, activation='none', pad_type=padding_type)
        )

    def forward(self, x):
        return x + self.block(x)


class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.scale_conv1 = nn.Conv2d(64, 64, 1)
        self.scale_conv2 = nn.Conv2d(64, 64, 1)
        self.shift_conv1 = nn.Conv2d(64, 64, 1)
        self.shift_conv2 = nn.Conv2d(64, 64, 1)

    def forward(self, x):
        cond = x[1]
        scale = self.scale_conv2(F.leaky_relu(self.scale_conv1(cond), 0.1, inplace=True))
        shift = self.shift_conv2(F.leaky_relu(self.shift_conv1(cond), 0.1, inplace=True))
        return x[0] * scale + shift

class ConvBlock_SFT(nn.Module):
    def __init__(self, dim, norm_type, padding_type, use_dropout=False):
        super(ConvBlock_SFT, self).__init__()
        self.sft1 = SFTLayer()
        self.conv1 = ConvBlock(dim, dim, 4, 2, 1, norm=norm_type, activation='none', pad_type=padding_type)

    def forward(self, x):
        fea = self.sft1((x[0], x[1]))
        fea = F.relu(self.conv1(fea), inplace=True)
        return (x[0] + fea, x[1])

class ConvBlock_SFT_last(nn.Module):
    def __init__(self, dim, norm_type, padding_type, use_dropout=False):
        super(ConvBlock_SFT_last, self).__init__()
        self.sft1 = SFTLayer()
        self.conv1 = ConvBlock(dim, dim, 4, 2, 1, norm=norm_type, activation='none', pad_type=padding_type)

    def forward(self, x):
        fea = self.sft1((x[0], x[1]))
        fea = F.relu(self.conv1(fea), inplace=True)
        return x[0] + fea

# Definition of normalization layer
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, \
            "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        mean = self.running_mean.repeat(b)
        var = self.running_var.repeat(b)
        x = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(x, mean, var, self.weight, self.bias, True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(1, 1, 1, 1)
            std = x.view(-1).std().view(1, 1, 1, 1)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(-1, 1, 1, 1)
            std = x.view(x.size(0), -1).std(1).view(-1, 1, 1, 1)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            x = x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0]

        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(w.view(height, -1).t().data, u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module(*args)