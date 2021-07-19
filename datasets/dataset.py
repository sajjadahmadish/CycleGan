import os
from datasets.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms


class M2PDataset:

    def __init__(self, opt, phase):
        self.opt = opt
        self.root = opt.dataset
        self.dir_A = os.path.join(opt.dataset, phase + 'A') 
        self.dir_B = os.path.join(opt.dataset, phase + 'B')  

        self.A_paths = sorted(make_dataset(self.dir_A))   
        self.B_paths = sorted(make_dataset(self.dir_B))   
        self.A_size = len(self.A_paths)  
        self.B_size = len(self.B_paths)  
        self.transform_A = get_transform()
        self.transform_B = get_transform()

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  
        index_B = index % self.B_size
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)


def get_transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
