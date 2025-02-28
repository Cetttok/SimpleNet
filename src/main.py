import torch
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

import numpy
import os
pathToSave = "../data/save"
imageSize = 8
coresSize = 3
coresCount = 32
hiddenLayerSize = 500
epochCount =30
learningSpeed = 0.001
godValue = 1
class FullConnectedNetwork:
    def __init__(self, firstLayerSize, hiddenSize,lastLayerSize):
        self.mWeights =  [Variable(torch.FloatTensor(hiddenSize,firstLayerSize).normal_(0.1), requires_grad = True), 
                      Variable(torch.FloatTensor(lastLayerSize,hiddenSize).normal_(0.1), requires_grad = True)]
        self.mBias = [Variable(torch.FloatTensor(hiddenSize).normal_(0.1), requires_grad = True),
                  Variable(torch.FloatTensor(lastLayerSize).normal_(0.1), requires_grad = True)]  #uniform,normal
        self.mLastResult = None

    def forward(self, input):
        for i in range(len(self.mWeights)):
            input = (self.mWeights[i].mv(input).add_(self.mBias[i]))
            if (i == len(self.mWeights)-1):
                input = torch.softmax(input,dim =0)
            else:
                input = torch.tanh(input)
        
                
            #torch.tanh
            #input = nn.ReLU(input)
        self.mLastResult = input
        return self.mLastResult

    def learn(self, speed, correct):
        #loss = (correct - self.mLastResult)
        
        #self.mLastResult.backward(loss)
        
        loss = abs(correct - self.mLastResult).mean()
        loss.backward()
        with torch.no_grad(): 
            for i in range(len(self.mWeights)):
                self.mWeights[i].data -= speed * self.mWeights[i].grad
                self.mBias[i].data -= speed * self.mBias[i].grad
        for i in range(len(self.mWeights)):
            self.mWeights[i].grad.zero_()
            self.mBias[i].grad.zero_()

 
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
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=coresCount, kernel_size=coresSize)
        self.layer2 = nn.Linear(coresCount*((imageSize -round(coresSize/2))**2) , hiddenLayerSize)
        self.layer3 = nn.Linear(hiddenLayerSize, 10)
    def forward(self, input):
        input = F.tanh(torch.flatten(F.relu(self.layer1(input))))
        #print(input)
        input = F.tanh(self.layer2(input))
        
        input = F.softmax(self.layer3(input),dim=0)
        return input
    
def getMaxId(list):
    result = 0
    for i in range(1,len(list)):
    
        if list[i] > list[result]:
            result = i
    return result        
dirList = os.listdir("../data/64")

print(dirList)

dataImages = []
dataImagesOutputs = []
for imageName in dirList:
    dataImagesOutputs.append(torch.FloatTensor([0,0,0,0,0,0,0,0,0,0]))
    
    dataImagesOutputs[len(dataImagesOutputs)-1][int(imageName[len(imageName)-5])] = godValue
    dataImages.append((torch.FloatTensor(numpy.array(Image.open("../data/64/"+imageName).convert('L'))).unsqueeze(0)*-1 + 255) / 255)
    print(dataImages[len(dataImages)-1])
    
#print(dataImages[0])
net = FullConnectedNetwork(imageSize**2,hiddenLayerSize,10)

net2 = AnotherNet()
print(net2)
print(net2(dataImages[0]))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net2.parameters(), lr=learningSpeed)
#optimizer = torch.load("../data/save")
#print("optim. loaded!")
#torch.save(net2.state_dict(), pathToSave)
#model = AnotherNet()
#net2.load_state_dict(torch.load(pathToSave, weights_only=True))
#net2.eval()
for i in range(epochCount):
    
    print(str(i) + " epoch:")
    correct = 0
    for imageId in range(len(dataImages)):
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