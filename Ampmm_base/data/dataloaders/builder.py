from .samplers import RankSampler, RankBatchSampler
from torch.utils.data import DataLoader

def build_data_loader(dataset,batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader