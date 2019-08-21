# Identical copies of two AlexNet models
import torch
import torch.nn as nn
import copy 
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


class FullyConnected(nn.Module):

    def __init__(self, input_dim=28*28 , width=50, depth=3, num_classes=10, use_batch_norm=True, activation="relu"):
        super(FullyConnected, self).__init__()
        self.input_dim = input_dim 
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm

        #if use_batch_norm:
        #    self.bn = nn.BatchNorm2d(64)
        #else:
        #    self.bn = Identity()
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)           
        if activation == "elu":
            self.act = nn.ELU(inplace=True)    
        if activation == "tanh":
            self.act = nn.Tanh()    

              
        layers = self.get_layers()
        if self.use_batch_norm:
            self.fc = nn.Sequential(
                nn.Linear(self.input_dim, self.width, bias=False),
                nn.BatchNorm1d(self.width), 
                nn.ReLU(inplace=True),
                layers,
                nn.Linear(self.width, self.num_classes, bias=False)
            )
        else: 
            self.fc = nn.Sequential(
                nn.Linear(self.input_dim, self.width, bias=False), 
                nn.ReLU(inplace=True),
                layers,
                nn.Linear(self.width, self.num_classes, bias=False)
            )
 


    def get_layers(self):
        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            if self.use_batch_norm:
               layers.append(nn.BatchNorm1d(self.width))  
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x


# This is a copy from online repositories 
class AlexNet(nn.Module):

    def __init__(self, input_height=32, input_width=32, input_channels=3, ch=64, num_classes=1000, use_batch_norm=True):
        # ch is the scale factor for number of channels
        super(AlexNet, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.use_batch_norm = use_batch_norm

        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels=ch, kernel_size=4, stride=2, padding=2),
            self.batchnorm2d(ch, self.use_batch_norm),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=5, padding=2),
            self.batchnorm2d(ch, self.use_batch_norm),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            self.batchnorm2d(ch, self.use_batch_norm), 
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            self.batchnorm2d(ch, self.use_batch_norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            self.batchnorm2d(ch, self.use_batch_norm),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.size = self.get_size()
        print(self.size)
        a = torch.tensor(self.size).float()
        b = torch.tensor(2).float()
        self.width = int(a) * int(1 + torch.log(a) / torch.log(b))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.size, self.width),
            self.batchnorm1d(self.width, self.use_batch_norm), 
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.width, self.width),
            self.batchnorm1d(self.width, self.use_batch_norm),     
            nn.ReLU(inplace=True),
            nn.Linear(self.width, num_classes),
        )

    def batchnorm2d(self, width, use_batch_norm): 
        if use_batch_norm:
            return nn.BatchNorm2d(width) 
        else:
            return nn.Sequential()

    def batchnorm1d(self, width, use_batch_norm):
        if use_batch_norm:
            return nn.BatchNorm1d(width)
        else:
            return nn.Sequential() 

    def get_size(self):
        # hack to get the size for the FC layer...
        x = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        y = self.features(x)
        print(y.size())
        return y.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


#This is a copy from online repositories
class Lenet(nn.Module):

    def __init__(self):
        super(Lenet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



def alexnet(**kwargs):
    return AlexNet(**kwargs)


def fc(**kwargs):
    return FullyConnected(**kwargs)

def lenet(**kwargs):
    return Lenet(**kwargs) 

if __name__ == '__main__':
    # testing
    
    x = torch.randn(5, 1, 32, 32)
    net = FullyConnected(input_dim=32*32, width=123)
    print(net(x))

    x = torch.randn(5, 3, 32, 32).cuda()
    net = AlexNet(ch=128).cuda()
    print(net(x))
