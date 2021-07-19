import torch.utils.data
from datasets.dataset import M2PDataset


def create_dataset(opt, phase):
    data_loader = CustomDatasetDataLoader(opt, phase)
    return data_loader.load_data()


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, phase):
        self.opt = opt
        self.dataset = M2PDataset(opt, phase)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle= False,
            num_workers=int(opt.workers))

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for data in self.dataloader:
            yield data