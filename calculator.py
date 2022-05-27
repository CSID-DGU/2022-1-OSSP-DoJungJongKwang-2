from PIL import Image,ImageTk
import numpy as np
from numpy import *
from sympy import *
from scipy.integrate import quad
import torch
from for_test_V20 import for_test
import matplotlib.pyplot as plt

def imresize(im,sz):
	pil_im = Image.fromarray(im)
	return np.array(pil_im.resize(sz))

def resize( w_box, h_box, pil_image): 
	w, h = pil_image.size 
	f1 = 1.0*w_box/w 
	f2 = 1.0*h_box/h    
	factor = min([f1, f2])   
	width = int(w*factor)    
	height = int(h*factor)    
	return pil_image.resize((width, height), Image.ANTIALIAS) 
#/content/drive/MyDrive/2022-1-OSSP2-DoJungJongKwang-2-kdy/off_image_test/504_em_39_0.bmp
#/content/drive/MyDrive/2022-1-OSSP2-DoJungJongKwang-2-kdy/IMAGE/극한_01_0.bmp
#/content/drive/MyDrive/2022-1-OSSP2-DoJungJongKwang-2-kdy/IMAGE/적분_01_0.bmp
img_test = '/content/drive/MyDrive/2022-1-OSSP2-DoJungJongKwang-2-kdy/IMAGE/극한_02_0.bmp'
img_open = Image.open(img_test).convert('L')
img_open2 = torch.from_numpy(np.array(img_open)).type(torch.FloatTensor)
img_open2 = img_open2/255.0
img_open2 = img_open2.unsqueeze(0)
img_open2 = img_open2.unsqueeze(0)
attention, prediction_real, prediction_symp, shape, int_var = for_test(img_open2)
#, function_symp, a_val, b_val
global prediction_string
prediction_string = ''
print(prediction_string)
img_open = np.array(img_open)

for i in range(attention.shape[0]):
	if prediction_real[i] == '<eol>':
		continue
	else:
		prediction_string = prediction_string + prediction_real[i]
print('prediction is')
print(prediction_string)
print('')
print('symp is')
print(''.join(prediction_symp))
print('')
print(prediction_symp)

if shape == 0:
	sol = eval(''.join(prediction_symp))
	print('result is')
	print(sol)

elif shape == 1:
	pos = []
	for i in range(len(prediction_symp)):
		if prediction_symp[i] == ' lim ':
			pos.append(i)
	x_val = prediction_symp[1:pos[0]-1]#5
	function_symp = prediction_symp[pos[0]+1:]
	lim_var = symbols(prediction_symp[1])
	#print(''.join(function_symp))
	#print(lim_var)
	#print(x_val)
	exec(''.join(x_val))
	sol = limit(''.join(function_symp), lim_var, x)
	print('result is')
	print(sol)

elif shape == 2:
	pos1 = []
	pos2 = []
	for i in range(len(prediction_symp)):
		if prediction_symp[i] == ',':
			pos1.append(i)
	for i in range(len(prediction_symp)):
		if prediction_symp[i] == ' quad ':
			pos2.append(i)
	b_val = prediction_symp[1:pos1[0]-1]#3
	a_val = prediction_symp[pos1[0]+2:pos2[0]-1]#8
	function_symp = prediction_symp[pos2[0]+1:]
	#print(int_var)
	exec('f=lambda '+ ''.join(int_var) + ':' + ''.join(function_symp))
	#print(a_val)
	#print(b_val)
	exec(''.join(a_val))
	exec(''.join(b_val))
	sol = quad(f,b,a)
	print('result is')
	print(sol[0])

else:
	print('error')