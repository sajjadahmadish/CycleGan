import argparse
from datasets import create_dataset
from tqdm import tqdm
import torch
from generate_sample import show
from model import CGModel
import time
import os
import matplotlib.pyplot as plt


def print_losses(epoch, iters, losses, verbose = False):
    message = '(epoch: %d, iters: %d) ' % (epoch, iters)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    if verbose:
        print(message)  # print the message
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='./out', help='output folder')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--netG',type=str, default='unet', help="generator network")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--continue_train', type=bool, default=False, help='continue with pretrained?')
    
    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--momentum', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--direction', type=str, default='A->B', help='A->B or B->A')   

    opt = parser.parse_args()
    print(opt)

    epochs = opt.nepoch

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    assert(torch.cuda.is_available())

    phase='train'
    dataset = create_dataset(opt, phase=phase)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size, '\n')

    model = CGModel(opt, isTrain=True, device= device)
    print("model [%s] was created" % type(model).__name__)
    if opt.continue_train:
        model.load_pretrained()

    log_name = os.path.join(opt.outf, 'losses_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write(('='*10) + f' Training Loss ({now}) ' + ('='*10) + '\n')

    print('')
    print('Training ...')
    t_iter = 0
    for epoch in range(epochs):
        iter = 0
        bar = tqdm(total=len(dataset), desc= f'epoch {epoch+1}/{epochs}', ncols=100)
        for i, data in enumerate(dataset):

            model.fit(data)
            
            iter += opt.batchSize
            t_iter += opt.batchSize

            if t_iter % 200 == 0:
                losses = model.get_losses()
                print_losses(epoch, iter, losses)
                visuals = model.get_current_visuals()
                show(visuals) 
            
            bar.update(1)

        model.save_networks(epoch)




