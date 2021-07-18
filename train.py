import argparse
from datasets import create_dataset
from tqdm import tqdm



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='out', help='output folder')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--netG',type=str, default='unet', help="generator network")



    opt = parser.parse_args()
    print(opt)

    epochs = opt.nepoch

    dataset = create_dataset(opt, phase='train')
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    print('')
    print('Training ...')
    for epoch in tqdm(range(epochs), ncols = 110):
        for i in tqdm(range(100), desc= f'epoch {epoch+1}/{epochs}', ncols= 110):
            1+1
