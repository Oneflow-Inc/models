import oneflow
import torch
class_num = 10
batch_size = 4
label = oneflow.tensor([[6],
        [0],
        [3],
        [2]])
m_hot=oneflow.zeros(batch_size, class_num)
m_hot_1=oneflow.scatter(m_hot,1, label, 0.5).to("cuda")
print(m_hot_1)
m=oneflow.scatter(m_hot,1, label, 0.2).to("cuda")


index=oneflow.tensor([0,1,2,3])   
m_hot_1[index]-=m
print(m_hot_1)
#tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])




label =torch.tensor([[6],
        [0],
        [3],
        [2]])
index=torch.tensor([0,1,2,3])       
tensor1=torch.zeros(batch_size, class_num).scatter(1, label, 0.5)
tensor2=torch.zeros(batch_size, class_num).scatter(1, label, 0.2)
tensor1[index]-=tensor2
print(tensor1)