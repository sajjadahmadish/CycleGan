import argparse
from datasets import create_dataset
from tqdm import tqdm
import torch
from model import CGModel
import matplotlib.pyplot as plt
import numpy as np


def tensor_image(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy() 
        if image_numpy.shape[0] == 1: 
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 
    else:  
        image_numpy = input_image
    return image_numpy.astype(imtype)


def show(visuals, prefix = ''):
    if prefix != '':
        prefix = str(prefix) + '_'
    for label, image in visuals.items():
        image_numpy = tensor_image(image)
        plt.imsave(f'./img/{prefix}{label}.jpg', image_numpy)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--max_num', type=int, required=True, help='input batch size')
    parser.add_argument('--outf', type=str, default='./out', help='output folder')
    parser.add_argument('--resultdir', type=str, default='./results', help='output folder')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--direction', type=str, default='A->B', help='A->B or B->A')   
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    
    opt = parser.parse_args()
    print(opt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    assert(torch.cuda.is_available())

    phase='test'
    dataset = create_dataset(opt, phase=phase)
    dataset_size = len(dataset)   

    model = CGModel(opt, isTrain=False, device= device)
    print("model [%s] was created" % type(model).__name__)

    print('')
    bar = tqdm(total = min(len(dataset), opt.max_num))
    for i, data in enumerate(dataset):
        model.set_input(data)  
        model.test()    
        
        visuals = model.get_current_visuals()
        show(visuals, i)

        bar.update(1)
        if i+1 == opt.max_num:
            break
            

        




