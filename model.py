import torch
import torch.nn as nn
from torch.nn import init
import functools
from collections import OrderedDict
import os
import random
import itertools



class CGModel():

    def __init__(self, opt, isTrain, device):
        
        self.opt = opt
        self.isTrain = isTrain
        self.device = torch.device(device)
        self.save_dir = os.path.join(opt.outf)
        self.loss_names = ['D_Y', 'G', 'cycle_A', 'idt_A', 'D_X', 'F', 'cycle_B', 'idt_B']
        self.model_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  

        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'idt_B', 'real_B', 'fake_A', 'rec_B', 'idt_A']

        self.model_names = ['G', 'F']
        if self.isTrain:
            self.model_names = ['G', 'F', 'D_Y', 'D_X']



        self.netG = init_weights(UnetGenerator(), self.device)
        self.netF = init_weights(UnetGenerator(), self.device)

        if isTrain:
            self.netD_Y = init_weights(PatchGANDiscriminator(), self.device)
            self.netD_X = init_weights(PatchGANDiscriminator(), self.device)


            self.fake_A_pool = ImagePool() 
            self.fake_B_pool = ImagePool() 
            # define loss functions
            self.criterionGAN = GANLoss().to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netF.parameters()), lr=opt.lr, betas=(opt.momentum, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_Y.parameters(), self.netD_X.parameters()), lr=opt.lr, betas=(opt.momentum, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    
    def set_input(self, input):
        AtoB = self.opt.direction == 'A->B'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)  
        self.rec_A = self.netF(self.fake_B)   
        self.fake_A = self.netF(self.real_B)  
        self.rec_B = self.netG(self.fake_A)   

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_Y(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_Y = self.backward_D_basic(self.netD_Y, self.real_B, fake_B)

    def backward_D_X(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_X = self.backward_D_basic(self.netD_X, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netF(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G = self.criterionGAN(self.netD_Y(self.fake_B), True)
        self.loss_F = self.criterionGAN(self.netD_X(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        self.loss_GAN = self.loss_G + self.loss_F + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_GAN.backward()

    def optimize_parameters(self):
        self.forward()      
        self.set_requires_grad([self.netD_Y, self.netD_X], False) 
        self.optimizer_G.zero_grad()  
        self.backward_G()             
        self.optimizer_G.step()       

        self.set_requires_grad([self.netD_Y, self.netD_X], True)
        self.optimizer_D.zero_grad()  
        self.backward_D_Y()      
        self.backward_D_X()      
        self.optimizer_D.step()  


    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  
        return errors_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                torch.save(net.module.cpu().state_dict(), save_path)
                net.cuda(0)


    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad



def init_weights(net, device, init_gain=0.02 ):
    net.to(device)
    net = torch.nn.DataParallel(net, [0])
    def init_func(m): 
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    net.apply(init_func)  
    return net

class GANLoss(nn.Module):

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class UnetGenerator(nn.Module):

    def __init__(self):
        num_downs = 8
        input_nc = 3
        output_nc = 3
        ngf=64
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  
        for i in range(num_downs - 5):        
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer) 

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d):

        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   
            return torch.cat([x, self.model(x)], 1)



class PatchGANDiscriminator(nn.Module):

    def __init__(self):

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        ndf=64
        n_layers = 3
        super(PatchGANDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial: 
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(3, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)



class ImagePool():
    def __init__(self, pool_size = 50):
        self.pool_size = pool_size
        if self.pool_size > 0: 
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:  
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5: 
                    random_id = random.randint(0, self.pool_size - 1) 
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:      
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)  
        return return_images