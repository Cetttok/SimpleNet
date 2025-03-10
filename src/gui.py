from tkinter import *
import time
import torch
from torch.autograd import Variable
from PIL import Image, ImageGrab
import torch.nn as nn
import io
import numpy
import torch.nn.functional as F
global mouseBtn1
mouseBtn1 = True
imageFileName = "image"
pixelsCount = 28
pixelIndent = 0
pixelNullColor = "#2F4F4F"
pixelBorderColor = "#000000"

fullValue = 1

pixelColoredColor = "#FFFFFF"
backgroundColor = "black"







pathToSave = "../data/save"

imageSize = 28
coresSize = 3
coresCount = 32
hiddenLayerSize = 100
epochCount =100
learningSpeed = 0.001
godValue = 1





def onResize(event):
    global canvas, penRadX, penRadY, root
 #   canvas.destroy()
    print("OnResize")
  #  canvas = Canvas(bg=backgroundColor, width=int(root.winfo_width()), height=int(root.winfo_height()))
    penRadX = round(int((root.winfo_width())/imageSize)*1.5)
    penRadY = round(int((root.winfo_height())/imageSize)*1.5)
    reDraw(canvas)
    canvas.config(width=int(root.winfo_width()), height=int(root.winfo_height()))
    #canvas.pack()
    
def onClose():
    print("exititng...")
    root.destroy()
    

root = Tk()    
root.geometry("250x200")
root.title("SimleNet GUI Tester")     
#root.geometry("300x250")    
root.protocol("WM_DELETE_WINDOW",onClose)
root.update()
print(root.size())
print(int(root.winfo_height()))
print(int(root.winfo_width()))
penRadX = round(int((root.winfo_width())/imageSize)*1.5)
penRadY = round(int((root.winfo_height())/imageSize)*1.5)
canvas = Canvas(bg=backgroundColor, width=int(root.winfo_width()), height=int(root.winfo_height()))
root.bind("<BackSpace>", onResize)
canvas.pack()
    

def reDraw(event):
    canvas.create_rectangle(0,0,int(root.winfo_width()),int(root.winfo_height()),fill=backgroundColor)




def onMousePress(event):
    print("pressed")
    global mouseBtn1, canvas 
    mouseBtn1 = True  
    canvas.create_oval(event.x- penRadX, event.y - penRadY, event.x+penRadX, event.y+penRadY, fill = pixelColoredColor, outline = pixelColoredColor)
        
    
def getPixels(fromCanvas):
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    img = ImageGrab.grab(bbox=(x, y, x1, y1))
    img = img.resize((imageSize,imageSize)).convert('L')
    
    img.save(imageFileName + ".png")

    return (torch.FloatTensor(numpy.array(img)).unsqueeze(0) / 255)
def onMouseRelease(event):
    global mouseBtn1
    print("released")
    mouseBtn1 = False
def onMouseMove(event):
    global mouseBtn1
    if mouseBtn1:
        canvas.create_oval(event.x- penRadX, event.y - penRadY, event.x+penRadX, event.y+penRadY, fill = pixelColoredColor, outline = pixelColoredColor)

    
canvas.bind("<ButtonPress-1>", onMousePress)
canvas.bind("<Motion>", onMouseMove)
canvas.bind("<ButtonRelease-1>", onMouseRelease)


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
    
    
    
    
    
    

    

    
def getMaxId(list):
    result = 0
    for i in range(1,len(list)):
    
        if list[i] > list[result]:
            result = i
    return result        
#print(dataImages[0])
net2 = AnotherNet()
#criterion = nn.L1Loss()
#optimizer = torch.optim.SGD(net2.parameters(), lr=learningSpeed, momentum=0.9)
#optimizer = torch.load("../data/save")
#print("optim. loaded!")
#torch.save(net2.state_dict(), pathToSave)
#model = AnotherNet()
net2.load_state_dict(torch.load(pathToSave, weights_only=True))
net2.eval()
list = [2,2,1,2,3,5,5,6,7]
print ("FUK - " + str(getMaxId(list)))


#root.bind("<BackSpace>",reDraw)

def getMaxId(list):
    result = 0
    for i in range(1,len(list)):
    
        if list[i] > list[result]:
            result = i
    return result        

def onReturn(event):
    #saveAsPng(canvas, imageFileName)
    print("Result:")
    result = (net2.forward(getPixels(canvas))) 
    #print(len(getPixels(canvas)))
    print(result)
    print(torch.argmax(result))
    print("__________")
root.bind("<Return>", onReturn)


mouseBtn1 = False

#canvas.create_line(10,10,200,200)

root.mainloop()