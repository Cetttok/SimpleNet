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
learningSpeed = 0.01
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
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.relu1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(720, 50)
        self.fc2 = nn.Linear(50, 10)        
    def forward(self, input):
        input = self.relu1(self.conv1(input))
        input = self.pool1(input)
        input  = torch.flatten(input)
        input  = torch.tanh(self.fc1(input))
        input = self.fc2(input)
        return torch.softmax(input, dim = 0)
def test(images, correct, net):
    correctCount = 0
    
    for i in range(len(images)):
        if getMaxId(net.forward(images[i])) == getMaxId(correct[i]):
            correctCount += 1
    return correctCount/len(images)
def getMaxId(list):
    result = 0
    for i in range(1,len(list)):
    
        if list[i] > list[result]:
            result = i
    return result        
print("START!")
net2 = AnotherNet()
print(net2)
#print(net2(dataImages[0]))
criterion = nn.NLLLoss()()
optimizer = torch.optim.SGD(net2.parameters(), lr=learningSpeed)

net2.load_state_dict(torch.load(pathToSave, weights_only=True))
net2.eval()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    print("!!!!!BAD-GPU!!!!!")
    device = torch.device("cpu")
net2.to(device)    
    
dirList = os.listdir("../data/train")
dirTestList = os.listdir("../data/test")
#print(dirList)
print("StartImage Loading: learn")
dataImages = []
dataImagesOutputs = []
for imageName in dirList:
    dataImagesOutputs.append(torch.cuda.FloatTensor([0,0,0,0,0,0,0,0,0,0]))
    dataImagesOutputs[len(dataImagesOutputs)-1][int(imageName[0])] = godValue
   # print("imageName: " + imageName + "output: " + str(list(dataImagesOutputs[len(dataImagesOutputs)-1]))) 
    
    
    dataImages.append(((torch.cuda.FloatTensor(numpy.array(Image.open("../data/train/"+imageName).convert('L'))).unsqueeze(0)) / 255))
    #print(dataImages[len(dataImages)-1])
print("StartImage Loading: test")

dataTestImages = []
dataTestOutputs = []



for imageName in dirTestList:
    dataTestOutputs.append(torch.cuda.FloatTensor([0,0,0,0,0,0,0,0,0,0]))
    dataTestOutputs[len(dataTestOutputs)-1][int(imageName[0])] = godValue
    dataTestImages.append(((torch.cuda.FloatTensor(numpy.array(Image.open("../data/test/"+imageName).convert('L'))).unsqueeze(0)) / 255))

print("imagesLoaded!")    
#print(dataImages[0])
#net = FullConnectedNetwork(imageSize**2,hiddenLayerSize,10)
#print(dataImages[0])

print("Start testing")
#print("testAcc = " + str(test(dataImages, dataImagesOutputs, net2)))


print("Training Started")
for i in range(epochCount):
    
    print(str(i) + " epoch:")
    correct = 0
    for imageId in range(len(dataImages)):
        if imageId%1000 == 0:
            print("image: " + str(imageId))
        data, target = Variable(dataImages[imageId]), Variable(dataImagesOutputs[imageId])
        optimizer.zero_grad()
        net_out = net2(data)
        if getMaxId(net_out) == getMaxId(dataImagesOutputs[imageId]):
            correct+= 1          
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()    
    print("last loss - " + str(loss))
    if correct/len(dataImages) > 0.99:
        print("mice -- " + str(correct/len(dataImages)))
        
        break
    print("acc = " + str(correct/len(dataImages)))
    print("testAcc = " + str(test(dataTestImages, dataTestOutputs, net2)))
    print("-------- ")
    print("")
#print(net.mWeights[0])
torch.save(net2.state_dict(), pathToSave)
print("saved")