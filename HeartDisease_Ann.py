import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd 

#Dataset Import
df = pd.read_excel("HeartDisease_new.xlsx")

x = df.drop(columns=df.columns[-1], axis =1).to_numpy()
y = df.iloc[:,-1].to_numpy()

sc = StandardScaler()
x = sc.fit_transform(x)

class dataset(Dataset):
  def __init__(self, x, y):
    self.x = torch.tensor(x, dtype = torch.float32)
    self.y = torch.tensor(y, dtype = torch.float32)
    self.length = self.x.shape[0]


  def __getitem__(self,idx):
    return self.x[idx], self.y[idx]

  def __len__(self):
    return self.length

trainset = dataset(x, y)
train_loader = DataLoader(trainset, batch_size = 64, shuffle= True)
#Defining the network

from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
  def __init__(self, input_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape, 32)
    self.fc2 = nn.Linear(32, 64)
    self.fc3 = nn.Linear(64,1)

  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))

#Hyper parameters
learning_rate = 0.02
epochs = 1000

#Model, optimizer, Loss
model = Net(input_shape=x.shape[1])
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
loss_fn = nn.BCELoss()

losses = []
accur = []

for i in range(epochs):
  for j, (x_train,y_train) in enumerate(train_loader):

    #Calculate output
    output = model(x_train)

    #Calculate loss
    loss = loss_fn(output, y_train.reshape(-1,1))

    #accuracy
    predicted = model(torch.tensor(x, dtype = torch.float32))
    acc = (predicted.reshape(-1).detach().numpy().round() == y).mean()

    #backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if i % 100 == 0 :
      losses.append(loss)
      accur.append(acc)
      print(f"epoch {i} \t Loss : {loss} \t accuracy : {acc}")