import argparse
from datasets import create_dataset
from tqdm import tqdm
import torch
import datasets
from model import CGModel



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='out', help='output folder')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--netG',type=str, default='unet', help="generator network")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    
    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--momentum', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')   

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

    print('')
    print('Training ...')
    for epoch in tqdm(range(epochs)):
        bar = tqdm(dataset, desc= f'epoch {epoch+1}/{epochs}')
        for i, data in enumerate(dataset):
            bar.update(1)
            bar.postfix = 'CC_loss: {}, Adv_loss: {}, identity_loss: {}'
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()
        
        print('saving the model at the end of epoch %d' % (epoch))
        model.save_networks('latest')
        model.save_networks(epoch)

