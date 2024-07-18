import torch
import torch.nn.functional as F


class VNet(torch.nn.Module):
    def __init__(self,input_dim,output_dim,lr=0.004,device=torch.device("cpu")):
        super(VNet,self).__init__()
        self.device=device

        self.conv1=torch.nn.Conv2d(input_dim,16,3,padding=1)
        self.conv2=torch.nn.Conv2d(16,16,8,4)
        self.conv3=torch.nn.Conv2d(16,32,4,2)
        self.fc1=torch.nn.Linear(32*9*9,256)
        self.fc2=torch.nn.Linear(256,output_dim)

        self.to(device)
        self.criterion=torch.nn.SmoothL1Loss()
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)

    def forward(self,x):
        out=F.relu(self.conv1(x))
        out=F.relu(self.conv2(out))
        out=F.relu(self.conv3(out))
        out=out.view(out.size(0),-1)

        out=F.relu((self.fc1(out)))
        out=self.fc2(out)
        return out

class CQNet(torch.nn.Module):
    def __init__(self,input_dim,action_dim,lr=0.001,device=torch.device("cpu")):
        super(CQNet,self).__init__()

        self.device=device

        self.conv1=torch.nn.Conv2d(input_dim,16,3,padding=1)
        self.conv2=torch.nn.Conv2d(16,16,8,stride=4)
        self.conv3=torch.nn.Conv2d(16,32,4,stride=2)

        self.fc1=torch.nn.Linear(32*9*9,256)
        self.fc2=torch.nn.Linear(256,action_dim)

        self.to(device)
        self.criterion=torch.nn.SmoothL1Loss()
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)

    def forward(self,x):
        out=F.relu(self.conv1(x))
        out=F.relu(self.conv2(out))
        out=F.relu(self.conv3(out))

        out=out.view(out.size(0),-1)
        out=F.relu(self.fc1(out))
        out=self.fc2(out)

        return out
