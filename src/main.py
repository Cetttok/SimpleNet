import torch
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

import numpy
import os
pathToSave = "../data/save"
imageSize = 28
coresSize = 3
coresCount = 8
hiddenLayerSize = 100
epochCount =4
learningSpeed = 0.001
godValue = 1

#net.mWeights = Variable(torch.FloatTensor([[1],[1],[1]]).t_(),requires_grad = True)
#print(net.forward(input))


#for i in range(0,1):
#    print("---------------")
#    for ii in range(3):
#        input = torch.FloatTensor(inputs[ii])
#        #print("result: "+str(float( )))
#        print(float(net.forward(input)))
#        net.learn(0.1,outputs[ii][0])

#print("weights:" + str(net.mWeights) + "\nbias:"+str(net.mBias))
class AnotherNet(nn.Module):
    def __init__(self):
        super(AnotherNet, self).__init__()
        #self.layer1 = nn.Conv2d(in_channels=1, out_channels=coresCount, kernel_size=coresSize)
        self.layer1 = nn.Linear(imageSize*imageSize, hiddenLayerSize)
        self.layer2 = nn.Linear(hiddenLayerSize , hiddenLayerSize)
        self.layer3 = nn.Linear(hiddenLayerSize, 10)
    def forward(self, input):
        input = F.relu(self.layer1(input))
        #print(input)
        input = F.relu(self.layer2(input))
        
        input = F.softmax(self.layer3(input),dim=0)
        return input
    
def getMaxId(list):
    result = 0
    for i in range(1,len(list)):
    
        if list[i] > list[result]:
            result = i
    return result        
dirList = os.listdir("../data/train")

print(dirList)

dataImages = []
dataImagesOutputs = []
for imageName in dirList:
    dataImagesOutputs.append(torch.FloatTensor([0,0,0,0,0,0,0,0,0,0]))
    
    dataImagesOutputs[len(dataImagesOutputs)-1][int(imageName[0])] = godValue
    dataImages.append(torch.flatten((torch.FloatTensor(numpy.array(Image.open("../data/train/"+imageName).convert('L'))).unsqueeze(0)) / 255))
    #print(dataImages[len(dataImages)-1])
    
#print(dataImages[0])
#net = FullConnectedNetwork(imageSize**2,hiddenLayerSize,10)
print(dataImages[0])
net2 = AnotherNet()
print(net2)
print(net2(dataImages[0]))
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net2.parameters(), lr=learningSpeed, momentum = 0.9)
#optimizer = torch.load("../data/save")
#print("optim. loaded!")
#torch.save(net2.state_dict(), pathToSave)
#model = AnotherNet()
net2.load_state_dict(torch.load(pathToSave, weights_only=True))
net2.eval()
for i in range(epochCount):
    
    print(str(i) + " epoch:")
    correct = 0
    for imageId in range(len(dataImages)):
        #if imageId%1000 == 0:
            #print("image: " + str(imageId))
        data, target = Variable(dataImages[imageId]), Variable(dataImagesOutputs[imageId])
        optimizer.zero_grad()
        net_out = net2(data)
        if getMaxId(net_out) == getMaxId(dataImagesOutputs[imageId]):
            correct+= 1          
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()    
    print("loss - " + str(loss))
    if correct/len(dataImages) > 0.99:
        print("mice -- " + str(correct/len(dataImages)))
        
        break
    print("acc = " + str(correct/len(dataImages)))
    print("-------- ")
    print("")
#print(net.mWeights[0])
torch.save(net2.state_dict(), pathToSave)
print("saved")