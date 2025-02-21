import torch
from torch.autograd import Variable
from PIL import Image
import numpy
import os
imageSize = 8
hiddenLayerSize = 40
epochCount =10000
learningSpeed = 0.01
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
    print(int(imageName[len(imageName)-5]))
    dataImagesOutputs[len(dataImagesOutputs)-1][int(imageName[len(imageName)-5])] = godValue
    dataImages.append((torch.FloatTensor(numpy.array(Image.open("../data/64/"+imageName).convert('L')).ravel().tolist())*-1 + 255) / 255)
#print(dataImages[0])
net = FullConnectedNetwork(imageSize**2,hiddenLayerSize,10)



for i in range(epochCount):
    
    print(str(i) + " epoch:")
    print( str(dataImagesOutputs[3]) +" result:" +  str((net.forward(dataImages[3]))))
    correct = 0;    
    for imageId in range(len(dataImages)):
        if getMaxId(net.forward(dataImages[imageId]))== getMaxId(dataImagesOutputs[imageId]):
            correct+= 1  
        net.learn(learningSpeed,dataImagesOutputs[imageId])
    print("acc = " + str(correct/len(dataImages)))
    print("-------- ")
    print("")
print(net.mWeights[0])