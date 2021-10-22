# from torch.utils.data import Dataset
# import torch
# import torch
# import numpy as np


def issue1():
    import  torch as oneflow
    from torch.utils.data import Dataset,DataLoader
    

    class MyDataset(Dataset):
        def __init__(self):
            self.data = torch.tensor([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
            self.label = torch.Tensor([1,1,0,0])

        def __getitem__(self,index):
            return self.data[index],self.label[index]

        def __len__(self):
            return len(self.data)

    mydataset = MyDataset()
    sampler = torch.utils.data.SequentialSampler(mydataset)
    batch_sampler = torch.utils.data.BatchSampler(sampler,3,True)

    data_loader = torch.utils.data.DataLoader(mydataset,
                                            batch_sampler=batch_sampler,
                                            num_workers=2,
                                            )                                 

    t = iter(data_loader)


import torch

a = torch.rand(6)
b = torch.rand(2,3)
c = a.reshape(*b.size())
print(a.shape)
print(b.shape)
print(c.shape)