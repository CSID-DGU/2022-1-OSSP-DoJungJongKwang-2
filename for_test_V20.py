import sys
sys.path.append(r"../")
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy
import torch.utils.data as data
from data_iterator import dataIterator
from Attention_RNN import AttnDecoderRNN
from Densenet_torchvision import densenet121
from PIL import Image
from numpy import *

gpu=[0]
dictionaries=['/content/drive/MyDrive/2022-1-OSSP2-DoJungJongKwang-2-kdy/dictionary.txt']
hidden_size = 256
batch_size_t = 1
maxlen = 100

def load_dict(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
      w=l.strip().split()
      lexicon[w[0]]=int(w[1])
    print('total words/phones',len(lexicon))
    return lexicon

worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
  worddicts_r[vv] = kk


def for_test(x_t):

  h_mask_t = []
  w_mask_t = []
  encoder = densenet121()
  attn_decoder1 = AttnDecoderRNN(hidden_size,112,dropout_p=0.5)

  encoder = torch.nn.DataParallel(encoder, device_ids=gpu)
  attn_decoder1 = torch.nn.DataParallel(attn_decoder1, device_ids=gpu)
  encoder = encoder.cuda()
  attn_decoder1 = attn_decoder1.cuda()

  encoder.load_state_dict(torch.load('/content/drive/MyDrive/2022-1-OSSP2-DoJungJongKwang-2-kdy/model/encoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'))
  attn_decoder1.load_state_dict(torch.load('/content/drive/MyDrive/2022-1-OSSP2-DoJungJongKwang-2-kdy/model/attn_decoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'))

  encoder.eval()
  attn_decoder1.eval()

  x_t = Variable(x_t.cuda())
  x_mask = torch.ones(x_t.size()[0],x_t.size()[1],x_t.size()[2],x_t.size()[3]).cuda()
  print(x_t.size())
  print(x_mask.size())
  x_t = torch.cat((x_t,x_mask),dim=1)
  x_real_high = x_t.size()[2]
  x_real_width = x_t.size()[3]
  h_mask_t.append(int(x_real_high))
  w_mask_t.append(int(x_real_width))
  x_real = x_t[0][0].view(x_real_high,x_real_width)
  output_highfeature_t = encoder(x_t)

  x_mean_t = torch.mean(output_highfeature_t)
  x_mean_t = float(x_mean_t)
  output_area_t1 = output_highfeature_t.size()
  output_area_t = output_area_t1[3]
  dense_input = output_area_t1[2]

  decoder_input_t = torch.LongTensor([111]*batch_size_t)
  decoder_input_t = decoder_input_t.cuda()

  decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).cuda()
  # nn.init.xavier_uniform_(decoder_hidden_t)
  decoder_hidden_t = decoder_hidden_t * x_mean_t
  decoder_hidden_t = torch.tanh(decoder_hidden_t)

  prediction = torch.zeros(batch_size_t,maxlen)
  #label = torch.zeros(batch_size_t,maxlen)
  prediction_sub = []
  label_sub = []
  decoder_attention_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda()
  attention_sum_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda()
  decoder_attention_t_cat = []


  for i in range(maxlen):
    decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(decoder_input_t,
	                                                                                     decoder_hidden_t,
	                                                                                     output_highfeature_t,
	                                                                                     output_area_t,
	                                                                                     attention_sum_t,
	                                                                                     decoder_attention_t,dense_input,batch_size_t,h_mask_t,w_mask_t,gpu)

	    
    decoder_attention_t_cat.append(decoder_attention_t[0].data.cpu().numpy())
    topv,topi = torch.max(decoder_output,2)
    if torch.sum(topi)==0:
      break
    decoder_input_t = topi
    decoder_input_t = decoder_input_t.view(batch_size_t)

    # prediction
    prediction[:,i] = decoder_input_t


  k = numpy.array(decoder_attention_t_cat)
  x_real = numpy.array(x_real.cpu().data)

  prediction = prediction[0]

  prediction_real = []
  ######
  prediction_symp = []
  int_var = []
  shape = 0

  frac_on = [0,0,0,0,0]
  frac_num = -1
  num_on = 0
  var_on = 0
  lim_on = 0
  int_on = 0
  
  a_on = 0
  b_on = 0
  d_on = 0
  ######
  for ir in range(len(prediction)):
    if int(prediction[ir]) ==0:
      break
    prediction_real.append(worddicts_r[int(prediction[ir])])
  ######
    if int(prediction[ir]) == 1:
      prediction_symp.append('>=')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 2:
      if num_on == 1:
        prediction_symp.append('*sqrt')
      else:
        prediction_symp.append('sqrt')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 3:
      prediction_symp.append('<=')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 4: #/
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 5: #infity
      prediction_symp.append('oo')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 6: #(
      if num_on == 1:
        prediction_symp.append('*'+worddicts_r[int(prediction[ir])])
      else:
        prediction_symp.append(worddicts_r[int(prediction[ir])])
        num_on = 0
        var_on = 0
    elif int(prediction[ir]) == 7: #,
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 8: #0
      num_on = 1
      if var_on == 1:
        prediction_symp.append('*' + worddicts_r[int(prediction[ir])])
        var_on = 0
      else:
        prediction_symp.append(worddicts_r[int(prediction[ir])])
    elif int(prediction[ir]) == 9:
      prediction_symp.append('...')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 10: #8
      num_on = 1
      if var_on == 1:
        prediction_symp.append('*' + worddicts_r[int(prediction[ir])])
        var_on = 0
      else:
        prediction_symp.append(worddicts_r[int(prediction[ir])])
    elif int(prediction[ir]) == 11:
      if num_on == 1:
        prediction_symp.append('*(sigma)')
      else:
        prediction_symp.append('(sigma)')
        num_on = 1
    elif int(prediction[ir]) == 12: #<
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 13:
      prediction_symp.append('+-')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 14:
      if num_on == 1:
        prediction_symp.append('*log')
      else:
        prediction_symp.append('log')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 15:
      if num_on == 1:
        prediction_symp.append('*(pi)')
      else:
        prediction_symp.append('(pi)')
        num_on = 1
    elif int(prediction[ir]) == 19:
      prediction_symp.append('limits')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 24: #}
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 28:
      if num_on == 1:
        prediction_symp.append('*tan')
      else:
        prediction_symp.append('tan')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 31:
      if num_on == 1:
        prediction_symp.append('*(gamma)')
      else:
        prediction_symp.append('(gamma)')
        num_on = 1
    elif int(prediction[ir]) == 32: #{
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 33: #'
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 34: #+
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 35:
      if num_on == 1:
        prediction_symp.append('*(theta)')
      else:
        prediction_symp.append('(theta)')
        num_on = 1
    elif int(prediction[ir]) == 36:
      prediction_symp.append('for all')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 37: #3
      num_on = 1
      if var_on == 1:
        prediction_symp.append('*' + worddicts_r[int(prediction[ir])])
        var_on = 0
      else:
        prediction_symp.append(worddicts_r[int(prediction[ir])])
    elif int(prediction[ir]) == 38: #7
      num_on = 1
      if var_on == 1:
        prediction_symp.append('*' + worddicts_r[int(prediction[ir])])
        var_on = 0
      else:
        prediction_symp.append(worddicts_r[int(prediction[ir])])
    elif int(prediction[ir]) == 39:
      int_on = 1
      num_on = 0
      var_on = 0
      shape = 2
    elif int(prediction[ir]) == 40:
      if num_on == 1:
        prediction_symp.append('*sin')
      else:
        prediction_symp.append('sin')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 44:
      prediction_symp.append('...')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 46: #[
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 47: #_
      if lim_on == 0 and int_on == 0:
        prediction_symp.append(worddicts_r[int(prediction[ir])])
        num_on = 0
        var_on = 0
      elif int_on == 1:
        b_on = 1 
    elif int(prediction[ir]) == 50:
      prediction_symp.append('...')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 56:
      if num_on == 1:
        prediction_symp.append('*cos')
      else:
        prediction_symp.append('cos')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 57:
      if num_on == 1:
        prediction_symp.append('*(')
      else:
        prediction_symp.append('(')

      if frac_on[frac_num] != 0:
        frac_on[frac_num] += 1
        num_on = 0
      if lim_on != 0:
        lim_on += 1
        num_on = 0
      if int_on != 0:
        if b_on == 1:
          prediction_symp.append('b=')
          int_on += 1
          num_on = 0 
        if a_on == 1:
          prediction_symp.append('a=')
          int_on += 1
          num_on = 0 

    elif int(prediction[ir]) == 58:
      if num_on == 1:
        prediction_symp.append('*(delta)')
      else:
        prediction_symp.append('(delta)')
        num_on = 1
    elif int(prediction[ir]) == 59:
      prediction_symp.append('!=')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 60:
      prediction_symp.append(' in ')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 61:
      if num_on == 1:
        prediction_symp.append('*(alpha)')
      else:
        prediction_symp.append('(alpha)')
        num_on = 1
    elif int(prediction[ir]) == 62:
      prediction_symp.append('*')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 63:
      lim_on = 1
      num_on = 0
      var_on = 0
      shape = 1
    elif int(prediction[ir]) == 64: #.
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 65: #2
      num_on = 1
      if var_on == 1:
        prediction_symp.append('*' + worddicts_r[int(prediction[ir])])
        var_on = 0
      else:
        prediction_symp.append(worddicts_r[int(prediction[ir])])
    elif int(prediction[ir]) == 66: #6
      num_on = 1
      if var_on == 1:
        prediction_symp.append('*' + worddicts_r[int(prediction[ir])])
        var_on = 0
      else:
        prediction_symp.append(worddicts_r[int(prediction[ir])])
    elif int(prediction[ir]) == 67:
      if num_on == 1:
        prediction_symp.append('*(lambda)')
      else:
        prediction_symp.append('(lambda)')
        num_on = 1
    elif int(prediction[ir]) == 68: #4
      num_on = 1
      if var_on == 1:
        prediction_symp.append('*' + worddicts_r[int(prediction[ir])])
        var_on = 0
      else:
        prediction_symp.append(worddicts_r[int(prediction[ir])])
    elif int(prediction[ir]) == 71:
      prediction_symp.append('exists')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 75:
      if int_on == 1:
        a_on = 1
      else:
        prediction_symp.append('**')
        num_on = 0
        var_on = 0
    elif int(prediction[ir]) == 82: #>
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 83:
      frac_num += 1
      frac_on[frac_num] += 1
    elif int(prediction[ir]) == 84:
      if lim_on != 1:
        prediction_symp.append('=')
      else:
        prediction_symp.append('->')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 85:
      prediction_symp.append('/')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 86: #!
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 87:
      if num_on == 1:
        prediction_symp.append('*(phi)')
      else:
        prediction_symp.append('(phi)')
        num_on = 1
    elif int(prediction[ir]) == 88: #)
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 89: #-
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 90: #1
      num_on = 1
      if var_on == 1:
        prediction_symp.append('*' + worddicts_r[int(prediction[ir])])
        var_on = 0
      else:
        prediction_symp.append(worddicts_r[int(prediction[ir])])
    elif int(prediction[ir]) == 91: #5
      num_on = 1
      if var_on == 1:
        prediction_symp.append('*' + worddicts_r[int(prediction[ir])])
        var_on = 0
      else:
        prediction_symp.append(worddicts_r[int(prediction[ir])])
    elif int(prediction[ir]) == 92: #9
      num_on = 1
      if var_on == 1:
        prediction_symp.append('*' + worddicts_r[int(prediction[ir])])
        var_on = 0
      else:
        prediction_symp.append(worddicts_r[int(prediction[ir])])
    elif int(prediction[ir]) == 93: #=
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 96:
      if num_on == 1:
        prediction_symp.append('*(beta)')
      else:
        prediction_symp.append('(beta)')
        num_on = 1
    elif int(prediction[ir]) == 100: #]
      prediction_symp.append(worddicts_r[int(prediction[ir])])
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 101:
      if num_on == 1:
        prediction_symp.append('*(mu)')
      else:
        prediction_symp.append('(mu)')
      num_on = 1
    elif int(prediction[ir]) == 105:
      prediction_symp.append('sum')
      num_on = 0
      var_on = 0
    elif int(prediction[ir]) == 110:
      prediction_symp.append(')')

      if frac_on[frac_num] != 0:
        frac_on[frac_num] -= 1
        if frac_on[frac_num] == 1:
          prediction_symp.append('/')
          frac_on[frac_num] -= 1
          frac_num -= 1

      if lim_on != 0:
        lim_on -= 1
        if lim_on == 1:
          prediction_symp.append(' lim ')
          lim_on = 0

      if int_on != 0:
        int_on -= 1
        if int_on == 1 and b_on == 1:
          prediction_symp.append(',')
          b_on = 0
        if int_on == 1 and a_on == 1:
          prediction_symp.append(' quad ')
          a_on = 0

      num_on = 0
      var_on = 0
    else: #문자
      var_on = 1
      if d_on == 1:
        int_var.append(worddicts_r[int(prediction[ir])])
        int_on = 0
        d_on = 0
      else:
        if int(prediction[ir]) == 22:
          if int_on != 1:
            num_on = 0
            d_on = 1
          else:
            if num_on == 1:
              prediction_symp.append('*d')
            else:
              prediction_symp.append('d')
              num_on = 1
        else:
          if num_on == 1:
            prediction_symp.append('*'+worddicts_r[int(prediction[ir])])    
          else:
            prediction_symp.append(worddicts_r[int(prediction[ir])])
            num_on = 1
    #symp = sp.sympify(prediction_symp)
    ######
  prediction_real.append('<eol>')


  prediction_real_show = numpy.array(prediction_real)
  prediction_symp_show = numpy.array(prediction_symp)
  return k, prediction_real_show, prediction_symp_show, shape, int_var