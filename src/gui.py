from tkinter import *
import time
import torch
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
global mouseBtn1
mouseBtn1 = True

pixelsCount = 8
pixelIndent = 0
pixelNullColor = "#2F4F4F"
pixelBorderColor = "#000000"

fullValue = 1

pixelColoredColor = "#6A5ACD"
pixelUnColoredColor = "#A9A9A9"







pathToSave = "../data/save"

imageSize = 8
coresSize = 3
coresCount = 32
hiddenLayerSize = 500
epochCount =100
learningSpeed = 0.001
godValue = 1





def onClose():
    print("exititng...")
    root.destroy()
class Pixel:
    def __init__(self, x, y,width , height):
        self.width = width
        self.height = height
        self._x = x
        self._y =y
        self.state = False # flase - uncolored, true - colored. laziness to write enum
    def getCurrentColor(self):
        if self.state:
            return pixelColoredColor
        else:
            return pixelUnColoredColor
            
    def rotate(self):
        self.state = not self.state
    def rotate(self, value):
        if value == True:
            self.state = True
        else:
            self.state = False
    def draw(self,canvas):
        canvas.create_rectangle(self._x,self._y, 
                                self._x + self.width, self._y + self.height,
                                fill = self.getCurrentColor(), outline = pixelBorderColor
                                )
    
    def isPointInPixel(self, x, y):
        if x >self._x and x < (self._x+self.width) and y > self._y and y < (self._y + self.height):
            return True
        return False
    
    def x(self):
        return self._x
    
    def y(self):
        return self._y
    
def getPixelUnderMouse(pixelList, mouseX, mouseY):
    for pixel in pixelList:
        if pixel.isPointInPixel(mouseX,mouseY):
            return pixel
    return False

root = Tk()    
root.geometry("250x200")
root.title("SimleNet GUI Tester")     
#root.geometry("300x250")    
root.protocol("WM_DELETE_WINDOW",onClose)
root.update()
print(root.size())
print(int(root.winfo_height()))
print(int(root.winfo_width()))
canvas = Canvas(bg="white", width=int(root.winfo_width()), height=int(root.winfo_height()))
canvas.pack()
    

pixelWidth = int(int(root.winfo_width())/pixelsCount)
pixelHeight = int(int(root.winfo_height())/pixelsCount)
pixels = []
def reDraw(canvas, pixelList):
    canvas.create_rectangle(0,0,int(root.winfo_width()),int(root.winfo_height()),fill="white")
    for pixel in pixelList:
        pixel.draw(canvas)





def onMousePress(event):
    print("pressed")
    global mouseBtn1 
    mouseBtn1 = True  
    pixel = getPixelUnderMouse(pixels, event.x, event.y)
    if not pixel:
        return
    else:
        pixel.rotate(True)
        
    reDraw(canvas,pixels)
    
def onMouseRelease(event):
    global mouseBtn1
    print("released")
    mouseBtn1 = False
def onMouseMove(event):
    global mouseBtn1
    if mouseBtn1:
        pixel = getPixelUnderMouse(pixels, event.x, event.y)
        if not pixel:
            return
        else:
            pixel.rotate(True)
            
        reDraw(canvas,pixels)
def getArrayOfValues(pixels):
    result = torch.FloatTensor(1,imageSize,imageSize)
    
    for y in range(imageSize):
        for x in range(imageSize):
            result[0][y][x] = (float(pixels[y*imageSize+x].state)*fullValue)
        
    return result
        
def clearField(event):
    global pixels,canvas,pixels
    print(getArrayOfValues(pixels))
    for pixel in pixels:
        pixel.rotate(False)
    reDraw(canvas,pixels)
canvas.bind("<ButtonPress-1>", onMousePress)
canvas.bind("<Motion>", onMouseMove)
canvas.bind("<ButtonRelease-1>", onMouseRelease)


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



root.bind("<BackSpace>",clearField)

def getMaxId(list):
    result = 0
    for i in range(1,len(list)):
    
        if list[i] > list[result]:
            result = i
    return result        

def onReturn(event):
    print("Result:")
    print(getMaxId(net2.forward(getArrayOfValues(pixels))))
    print("__________")
root.bind("<Return>", onReturn)


mouseBtn1 = True
for y in range(pixelsCount):
    for x in range(pixelsCount):
        pixels.append(Pixel(x*pixelWidth + pixelIndent, y*pixelHeight + pixelIndent, 
                                pixelWidth - pixelIndent, pixelHeight - pixelIndent))
        #time.sleep(1)
        pixels[len(pixels)-1].draw(canvas)
        #root.update()
    
#canvas.create_line(10,10,200,200)

root.mainloop()