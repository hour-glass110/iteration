from math import pi
from matplotlib import pyplot as plt
import numpy as np
import torch
from net_2d import *
from nolinear import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpudir")
print(device)
# train_dataset = MyDataset(data_path="jihuo/",data_name="data6.txt",data_lable="label6.txt",transform=transforms.ToTensor())
x = np.loadtxt("data2d_4_withempty(1).txt",delimiter=" ")
x = torch.from_numpy(x).reshape(-1,1).float()
x = x.to(device)
print(x.is_cuda)
y = np.loadtxt("label2d_4_withempty(1).txt",delimiter=" ")
y = torch.from_numpy(y).reshape(-1,1).float()
y =y.to(device)
print(y.is_cuda)

#
net = nolinear().to(device)
net.load_state_dict(torch.load("data1d_12.pth"))

loss_fn = torch.nn.MSELoss()
loss_fn.to(device)
#创建优化器
learning_rata =  1e-4 / 3 #或者使用1e-2代替0.01
# optimizer = torch.optim.SGD(net.parameters(),lr=learning_rata)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rata)

#训练的次数
total_train_step = 0
#测试次数
total_test_step = 0
#训练轮数
R =50

# c = torch.from_numpy(np.linspace(0, 1, num=640)).reshape(-1, 1).float()
# print(c)
# d = net(c)

# writer = SummaryWriter("log_maxernet")
# start_time = time.time()
train_num = 0
num_of_true = 0
count =0

# z = torch.tensor([[1/20]])

# y=net(z)

# print(y)



for t in range(30000):
    # Forward pass: compute predicted y by passing x to the model.
    # for name,para in net.named_parameters():
    #     print(para)
    y_pred = net(x)
    # print("\n")
    
    # Compute loss and print it periodically
    loss = loss_fn(y_pred, y)
    
    if t % 100 == 0:
        print(t, loss.item())
    
    # if t % stepnum == 0:
    #     print(scheduler.get_lr())
        
 
    # Update the network weights using gradient of the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()
if(loss.item()<=0.25):
    # torch.save(net,"data2d_4(2).pth")
    torch.save(net.state_dict(),"data1d_12.pth")
 
# Draw the original random points as a scatter plot
plt.figure()
# fig.add_trace(plt.Scatter(x=x.flatten().numpy(), y=y.flatten().numpy(), mode="markers"))
 
# Generate predictions for evenly spaced x-values between minx and maxx
minx = min(list(x.cpu().numpy()))
maxx = max(list(x.cpu().numpy()))
c = torch.from_numpy(np.linspace(minx, maxx, num=640)).reshape(-1, 1).float().to(device)
d = net(c)
 
# Draw the predicted functions as a line graph
plt.title(label = "nolinear")
plt.scatter(x=x.cpu().flatten().numpy(), y=y.cpu().flatten().numpy(), c="r",marker=".")
# plt.scatter(x=c.flatten().numpy(), y=d.flatten().detach().numpy(), c="b",marker=".")
plt.plot(c.cpu().flatten().numpy(), d.cpu().flatten().detach().numpy())
plt.show()