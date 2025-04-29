import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

def generate_discrete_label(inputs, label_nc, device):
    pred_batch = []
    size = inputs.size()
    for input in inputs:
        input = input.view(1, label_nc, size[2], size[3])
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch).long().to(device)
    label_map = pred_batch.view(-1, 1, size[2], size[3])

    oneHot_size = (label_map.size(0), label_nc, size[2], size[3])
    input_label = torch.zeros(oneHot_size, device=device)
    input_label.scatter_(1, label_map, 1.0)
    
    return input_label

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, use_gan_feat_loss, use_vgg_loss, True, True, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, gb_gan, gb_gan_feat, gb_vgg, d_real, d_fake, d_blend):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,gb_gan,gb_gan_feat,gb_vgg,d_real,d_fake,d_blend),flags) if f]
        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.device = opt.device
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        if opt.resize_or_crop != 'none' or not opt.isTrain:
            torch.backends.cudnn.benchmark = True

        # Define Generator
        self.netG = networks.define_G(
            input_nc, opt.output_nc, opt.ngf, opt.netG,
            opt.n_downsample_global, opt.n_blocks_global,
            opt.n_local_enhancers, opt.n_blocks_local,
            opt.norm, gpu_ids=self.gpu_ids
        )

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            netB_input_nc = opt.output_nc * 2

            self.netD = networks.define_D(
                netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                use_sigmoid, opt.num_D, not opt.no_ganFeat_loss,
                gpu_ids=self.gpu_ids
            )

            self.netB = networks.define_B(
                netB_input_nc, opt.output_nc, 32, 3, 3,
                opt.norm, gpu_ids=self.gpu_ids
            )

        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # Load pre-trained models
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                self.load_network(self.netB, 'B', opt.which_epoch, pretrained_path)
        
        if self.isTrain:
            if opt.pool_size > 0 and len(self.gpu_ids) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            self.loss_filter = self.init_loss_filter(
                use_gan_feat_loss=not opt.no_ganFeat_loss,
                use_vgg_loss=not opt.no_vgg_loss
            )

            self.criterionGAN = networks.GANLoss(
                use_lsgan=not opt.no_lsgan,
                tensor=self.Tensor
            )

            self.criterionFeat = nn.L1Loss()

            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            self.loss_names = self.loss_filter(
                'G_GAN', 'G_GAN_Feat', 'G_VGG',
                'GB_GAN', 'GB_GAN_Feat', 'GB_VGG',
                'D_real', 'D_fake', 'D_blend'
            )

            # Optimizer for Generator (fine-tuning if requested)
            if opt.niter_fix_global > 0:
                finetune_list = set()
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params.append(value)
                        finetune_list.add(key.split('.')[0])
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are', sorted(finetune_list))
            else:
                params = list(self.netG.parameters())

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_GB = torch.optim.Adam(
                list(self.netG.parameters()) + list(self.netB.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
    
    def encode_input(self, inter_label_map_1, label_map, inter_label_map_2, real_image, label_map_ref, real_image_ref, infer=False):
        device = self.device

        if self.opt.label_nc == 0:
            input_label = label_map.to(device)
            inter_label_1 = inter_label_map_1.to(device)
            inter_label_2 = inter_label_map_2.to(device)
            input_label_ref = label_map_ref.to(device)
        else:
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.zeros(oneHot_size, device=device).scatter_(1, label_map.long().to(device), 1.0)
            inter_label_1 = torch.zeros(oneHot_size, device=device).scatter_(1, inter_label_map_1.long().to(device), 1.0)
            inter_label_2 = torch.zeros(oneHot_size, device=device).scatter_(1, inter_label_map_2.long().to(device), 1.0)
            input_label_ref = torch.zeros(oneHot_size, device=device).scatter_(1, label_map_ref.long().to(device), 1.0)

            if self.opt.data_type == 16:
                input_label = input_label.half()
                inter_label_1 = inter_label_1.half()
                inter_label_2 = inter_label_2.half()
                input_label_ref = input_label_ref.half()

        real_image = real_image.to(device)
        real_image_ref = real_image_ref.to(device)

        return inter_label_1, input_label, inter_label_2, real_image, input_label_ref, real_image_ref

    def encode_input_test(self, label_map, label_map_ref, real_image_ref, infer=False):
        device = self.device

        if self.opt.label_nc == 0:
            input_label = label_map.to(device)
            input_label_ref = label_map_ref.to(device)
        else:
            # Clamp to valid label range before one-hot encoding
            label_map = torch.clamp(label_map.long(), 0, self.opt.label_nc - 1)
            label_map_ref = torch.clamp(label_map_ref.long(), 0, self.opt.label_nc - 1)

            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.zeros(oneHot_size, device=device).scatter_(1, label_map.to(device), 1.0)
            input_label_ref = torch.zeros(oneHot_size, device=device).scatter_(1, label_map_ref.to(device), 1.0)

            if self.opt.data_type == 16:
                input_label = input_label.half()
                input_label_ref = input_label_ref.half()

        real_image_ref = real_image_ref.to(device)
        return input_label, input_label_ref, real_image_ref

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD(fake_query)
        else:
            return self.netD(input_concat)

    def forward(self, inter_label_1, label, inter_label_2, image, label_ref, image_ref, infer=False):
        inter_label_1, input_label, inter_label_2, real_image, input_label_ref, real_image_ref = self.encode_input(
            inter_label_1, label, inter_label_2, image, label_ref, image_ref, infer=infer
        )

        fake_inter_1 = self.netG(inter_label_1, input_label, real_image)
        fake_image = self.netG(input_label, input_label, real_image)
        fake_inter_2 = self.netG(inter_label_2, input_label, real_image)

        blend_image, alpha = self.netB(fake_inter_1, fake_inter_2)

        # Fake detection
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        pred_blend_pool = self.discriminate(input_label, blend_image, use_pool=True)
        loss_D_blend = self.criterionGAN(pred_blend_pool, False)

        # Real detection
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Generator loss
        pred_fake = self.netD(torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        pred_blend = self.netD(torch.cat((input_label, blend_image), dim=1))
        loss_GB_GAN = self.criterionGAN(pred_blend, True)

        # Feature matching loss
        loss_G_GAN_Feat = 0
        loss_GB_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                    loss_GB_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_blend[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG loss
        loss_G_VGG = 0
        loss_GB_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
            loss_GB_VGG = self.criterionVGG(blend_image, real_image) * self.opt.lambda_feat

        return [
            self.loss_filter(
                loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG,
                loss_GB_GAN, loss_GB_GAN_Feat, loss_GB_VGG,
                loss_D_real, loss_D_fake, loss_D_blend
            ),
            None if not infer else fake_inter_1,
            fake_image, fake_inter_2, blend_image,
            alpha, real_image, inter_label_1,
            input_label, inter_label_2
        ]

    def inference(self, label, label_ref, image_ref):
        with torch.no_grad():
            input_label, input_label_ref, real_image_ref = self.encode_input_test(
                label, label_ref, image_ref, infer=True
            )
            fake_image = self.netG(input_label, input_label_ref, real_image_ref)
        return fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch)
        self.save_network(self.netD, 'D', which_epoch)
        self.save_network(self.netB, 'B', which_epoch)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start fine-tuning it
        params = list(self.netG.parameters())
        if hasattr(self, 'gen_features') and self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also fine-tuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_GB.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print(f'update learning rate: {self.old_lr:.6f} -> {lr:.6f}')
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        return self.inference(inp, inp, inp)