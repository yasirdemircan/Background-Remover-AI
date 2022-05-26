from PIL import Image
import numpy
import os
import webbrowser
url = 'http://localhost:8000'
chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
""" 
image = Image.open("predimg.png").convert("RGBA")
imgwidth,imgheight = image.size
pixel_values_img = list(image.getdata())
pixel_values_img = numpy.array(pixel_values_img).reshape((imgwidth, imgheight, 4)) # getting imageas RGBA (4 channels)

mask = Image.open("mask.png").convert("RGB")
width, height = mask.size
pixel_values_mask = list(mask.getdata())
pixel_values_mask = numpy.array(pixel_values_mask).reshape((width, height, 3)) #Getting mask as rgb (3 channels)
bools = numpy.ones((width, height),dtype=numpy.bool) #2D Boolean array from the mask

for i in range(0,width):
    for k in range(0,height):
        if pixel_values_mask[i][k][0] == 127: #if starting R is gray (127,127,127)
            bools[i][k] = False
        else:
            bools[i][k] = True


for i in range(0,imgwidth):
    for k in range(0,imgheight):
        if bools[i][k] == False: #if starting R is gray (127,127,127)
            pixel_values_img[i][k][0] = 0
            pixel_values_img[i][k][1] = 0
            pixel_values_img[i][k][2] = 0
            pixel_values_img[i][k][3] = 0

        else:
            pixel_values_img[i][k][0] = 255
            pixel_values_img[i][k][1] = 255
            pixel_values_img[i][k][2] = 255
            pixel_values_img[i][k][3] = 255

print(bools,bools.shape)
image_out = Image.fromarray(pixel_values_img,"RGBA")
image_out.save("finalimg.png") """
webbrowser.get(chrome_path).open(url)