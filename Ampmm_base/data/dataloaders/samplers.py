import torch
from torch.utils.data.sampler import Sampler

class RankSampler(Sampler):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        indices = []
        for n in range(self.data.num_bac):
            index = torch.where(self.data.bacterium_id == n)[0]
            indices.append(index)
        indices = torch.cat(indices, dim=0)
        return iter(indices)

    def __len__(self):
        return len(self.data)

class RankBatchSampler:
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.epoch = 0
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        # print(len(sampler_list))
        for idx in sampler_list:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            if (
                i < len(sampler_list) - 1
                and self.sampler.data.bacterium_id[idx]
                != self.sampler.data.bacterium_id[sampler_list[i + 1]]
            ):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
                else:
                    batch = []
            i += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
        
    def set_epoch(self, epoch):
        self.epoch = epoch