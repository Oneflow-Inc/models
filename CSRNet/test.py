from matplotlib import pyplot as plt
from image import *
from model import CSRNet
import oneflow as flow
from matplotlib import cm as c
import transforms.spatial_transforms as ST

transform = ST.Compose([
                       ST.ToNumpyForVal(),ST.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

model = CSRNet()
model = model.to("cuda")
checkpoint = flow.load('checkpoint/Shanghai_BestModelA/shanghaiA_bestmodel')
model.load_state_dict(checkpoint)
img = transform(Image.open('dataset/part_A_final/test_data/images/IMG_100.jpg').convert('RGB'))
img=flow.Tensor(img)
img=img.to("cuda")
output = model(img.unsqueeze(0))
print("Predicted Count : ", int(output.detach().to("cpu").sum().numpy()))
temp = output.detach().reshape((output.detach().shape[2],output.detach().shape[3]))
temp=temp.numpy()
plt.title("Predicted Count")
plt.imshow(temp, cmap = c.jet)
plt.show()
temp = h5py.File('dataset/part_A_final/test_data/ground_truth/IMG_100.h5', 'r')
temp_1 = np.asarray(temp['density'])
plt.title("Original Count")
plt.imshow(temp_1,cmap = c.jet)
print("Original Count : ",int(np.sum(temp_1)) + 1)
plt.show()
print("Original Image")
plt.title("Original Image")
plt.imshow(plt.imread('dataset/part_A_final/test_data/images/IMG_100.jpg'))
plt.show()
