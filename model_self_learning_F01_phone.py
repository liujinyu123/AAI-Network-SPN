#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    Implementation with pytorch.
    Architecture : 2 dense layers, 2 bilstm layers (300 units), 1 dense layer (18 units), 1 conv layer
    Convolutional layer with weights so that it smooth the data at a cutoff frequency of 10Hz.
    Posibility to let the weights of the conv be updated during the training.
    Posibility to add batch normalization layer after the lstm layers
    [ future work : maybe batch norma after dense layers instead ?]
    Input of the nn : acoustic features for one sentence (K,429), K frames mfcc, 429 features per frame mfcc
    Ouput of the nn : articulatory trajectory for one sentence (K,18), one articulatory position per frame mfcc.
    Evaluation of the model with unseen sentences.
    Calculate the pearson and rmse between true and predicted traj for each sentence and average over sentences
    of the test set.

"""


import torch
import os,sys,inspect
from torch.autograd import Variable
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import math
import torch.nn as nn
import numpy as np
import gc
from Training.tools_learning import get_right_indexes, criterion_pearson_no_reduction
from Training.model_learning import ac2art_model
import random
import torch.nn.functional as F
from math import sqrt
class CNN_1D(torch.nn.Module):
    def __init__(self):
        super(CNN_1D,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=40,out_channels=128,kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
        self.conv5 = nn.Conv1d(in_channels=128,out_channels=128,kernel_size=5,padding=2)
        self.conv7 = nn.Conv1d(in_channels=128,out_channels=128,kernel_size=7,padding=3)
        self.conv9 = nn.Conv1d(in_channels=128,out_channels=128,kernel_size=9,padding=4)
        self.pool1 = nn.MaxPool1d(1)
        self.pool3 = nn.MaxPool1d(3,stride=1,padding=1)
        self.pool5 = nn.MaxPool1d(5,stride=1,padding=2)
        self.pool7 = nn.MaxPool1d(7,stride=1,padding=3)
        self.pool9 = nn.MaxPool1d(9,stride=1,padding=4)
        self.LeakyReLU = nn.LeakyReLU()
    def forward(self,x):
        x = self.pool1(self.LeakyReLU(self.conv1(x)))
        out = x
        x = self.pool3(self.LeakyReLU(self.conv3(x)))
        out = torch.cat((out,x),1)
        x = self.pool5(self.LeakyReLU(self.conv5(x)))
        out = torch.cat((out,x),1)
        x = self.pool7(self.LeakyReLU(self.conv7(x)))
        out = torch.cat((out,x),1)
        x = self.pool9(self.LeakyReLU(self.conv9(x)))
        out = torch.cat((out,x),1)
        return out
class phone_CNN_1D(torch.nn.Module):
    def __init__(self):
        super(phone_CNN_1D,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=61,out_channels=128,kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
        self.conv5 = nn.Conv1d(in_channels=128,out_channels=128,kernel_size=5,padding=2)
        self.conv7 = nn.Conv1d(in_channels=128,out_channels=128,kernel_size=7,padding=3)
        self.conv9 = nn.Conv1d(in_channels=128,out_channels=128,kernel_size=9,padding=4)
        self.pool1 = nn.MaxPool1d(1)
        self.pool3 = nn.MaxPool1d(3,stride=1,padding=1)
        self.pool5 = nn.MaxPool1d(5,stride=1,padding=2)
        self.pool7 = nn.MaxPool1d(7,stride=1,padding=3)
        self.pool9 = nn.MaxPool1d(9,stride=1,padding=4)
        self.LeakyReLU = nn.LeakyReLU()
    def forward(self,x):
        x = self.pool1(self.LeakyReLU(self.conv1(x)))
        out = x
        x = self.pool3(self.LeakyReLU(self.conv3(x)))
        out = torch.cat((out,x),1)
        x = self.pool5(self.LeakyReLU(self.conv5(x)))
        out = torch.cat((out,x),1)
        x = self.pool7(self.LeakyReLU(self.conv7(x)))
        out = torch.cat((out,x),1)
        x = self.pool9(self.LeakyReLU(self.conv9(x)))
        out = torch.cat((out,x),1)
        return out


def memReport(all = False):
    """
    :param all: wheter to detail all size obj
    :return: n objects
    In case of memory troubles call this function
    """
    nb_object = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if all:
                print(type(obj), obj.size())
            nb_object += 1
    print('nb objects tensor', nb_object)

class Fusion(nn.Module):

    def __init__(self):
        super(Fusion, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(813,128),nn.LeakyReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(6,12),nn.LeakyReLU(True))
    def forward(self, x):
        a = self.fc1(x[:,:,0:813])
        b = self.fc2(x[:,:,853:859])
        return torch.cat((a,b),2)
class fus(nn.Module):

    def __init__(self):
        super(fus, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(813,300),nn.LeakyReLU(True))
        # self.fc2 = nn.Sequential(nn.Linear(40,20),nn.LeakyReLU(True))
    def forward(self, x):
        # print(x.shape)
        a = self.fc1(x[:,:,0:813])
        # b = self.fc2(x[:,:,813:853])
        # return torch.cat((a,b),2)
        return a;
class gen(nn.Module):

    def __init__(self, input_size=39, hidden_size=100, num_layer=3, output_size=6):
        super(gen, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.output_size = output_size
        # self.fusion = Fusion()
        self.lstm_net = nn.LSTM(419,150,num_layer, batch_first=True, bidirectional=True)
        self.layer1 = nn.Linear(300, 300)
        self.layer2 = nn.Linear(300, output_size)

    def forward(self, x):
        output, (final_hidden_state, final_cell_state) = self.lstm_net(x)
        # print(output.shape)
        return self.layer2(self.layer1(output))

class y_fu_ext(nn.Module):

    def __init__(self, input_size=300, hidden_size=150, num_layer=3, output_size=6):
        super(y_fu_ext, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.output_size = output_size
        # self.fus = fus()
        self.lstm_net = nn.LSTM(413,hidden_size,num_layer, batch_first=True, bidirectional=True)
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size*2),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_size*2, output_size)
    def forward(self, x):
        # print(x.shape)
        # x = self.fus(x)
        # print(x.shape)
        output, (final_hidden_state, final_cell_state) = self.lstm_net(x)
        return self.layer2(output)

class ext(nn.Module):

    def __init__(self, input_size=61, hidden_size=150, num_layer=3, output_size=6):
        super(ext, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.output_size = output_size
        self.lstm_net = nn.LSTM(input_size,hidden_size,num_layer, batch_first=True, bidirectional=True)
        # self.attention_layer = nn.Sequential(
        #     nn.Linear(self.hidden_size*2, self.hidden_size*2),
        #     nn.ReLU(inplace=True)
        # )
        self.layer1 = nn.Linear(hidden_size*2, 300)
        self.layer2 = nn.Linear(300, 12)
    def forward(self, x):
        output, (final_hidden_state, final_cell_state) = self.lstm_net(x)
        return self.layer2(self.layer1(output))
#多头注意力机制
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # self.layer = nn.Linear(512,512);
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.activate = nn.ReLU()

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # enc_output = self.layer_norm(enc_output+self.activate(enc_output))
        return enc_output, enc_slf_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # print(q.shape)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    # print(seq.size())
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=300):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # print(x.shape)
        # print(self.pos_table.shape)
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=300):

        super().__init__()

        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        # self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        # self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask=None, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        #print("aaa",src_seq.shape)
        
        # enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq)))
        # print("1",src_seq.shape)
        # enc_output = self.dropout(self.position_enc(src_seq))
        # print("2",enc_output.shape)
        #print("aaa",enc_output.shape)
        # enc_output = self.layer_norm(enc_output)
        enc_output = src_seq
        # print(enc_output.shape)
        for enc_layer in self.layer_stack:
            
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            # print("bbbenc",enc_output.shape)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k)
        
    
    def forward(self,x):
        Q = self.q(x) # Q: batch_size * seq_len * dim_k
        K = self.k(x) # K: batch_size * seq_len * dim_k
        V = self.v(x) # V: batch_size * seq_len * dim_v
         
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        
        output = torch.bmm(atten,V) # Q * K.T() * V # batch_size * seq_len * dim_v
        
        return output
class my_ac2art_model(torch.nn.Module):
    """
    pytorch implementation of neural network
    """
    def __init__(self, hidden_dim, input_dim, output_dim, batch_size,name_file="", sampling_rate=100,cutoff=10,cuda_avail =False, filter_type=1, batch_norma=False):
        """
        :param hidden_dim: int, hidden dimension of lstm (usually 300)
        :param input_dim: int, input dimension of the acoustic features for 1 frame mfcc (usually 429)
        :param output_dim: int, # of trajectories to predict (usually 18)
        :param batch_size:  int, usually 10
        :param name_file: str, name of the model
        :param sampling_rate: int, sampling rate of the ema data for the smoothing (usually 100)
        :param cutoff: int, intial cutoff frequency for the smoothing, usually 10Hz
        :param cuda_avail: bool, whether gpu is available
        :param filter type: str, "out": filter outside the nn, "fix" : weights are FIXED,
        "unfix" : weights are updated during the training
        :param batch_norma: bool, whether to add batch normalization after the lstm layers
        """
        super(my_ac2art_model, self).__init__()
        self.encoder = Encoder(
            n_src_vocab=0, n_position=300,
            d_word_vec=512, d_model=40, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64,
            pad_idx=0, dropout=0.1)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        #self.num_layer = num_layer
        self.extractor = y_fu_ext(413,hidden_dim,3,6)
        self.phone_extractor = ext(61, hidden_dim, 3, 12)
        self.generator = gen(413+6,hidden_dim,3,6)
        self.fc1 = nn.Sequential(
            nn.Linear(853,300),
            nn.LeakyReLU(True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256,300),
            nn.LeakyReLU(True),
            nn.Linear(300,300),
            nn.LeakyReLU(True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(320,320),
            nn.LeakyReLU(True),
            nn.Linear(320,100),
            nn.LeakyReLU(True)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(320,320),
            nn.LeakyReLU(True),
            nn.Linear(320,100),
            nn.LeakyReLU(True)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(512,100),
            nn.LeakyReLU(True)
        )
        self.CNN_1D = CNN_1D()
        # self.phone_feature_extraction = phone_feature_extraction()
        self.layer1 = nn.Linear(101,512,bias=False)
        self.layer2 = nn.Linear(512,40,bias=False)
        #####
        
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.filter_type=filter_type
        self.softmax = torch.nn.Softmax(dim=output_dim)
        self.tanh = torch.nn.Tanh()
        self.sampling_rate = sampling_rate
        self.cutoff = cutoff
        self.N = None
        self.min_valid_error = 100000
        self.all_training_loss = []
        self.all_validation_loss = []
        self.all_test_loss = []
        self.name_file = name_file
        self.lowpass = None
        self.init_filter_layer()
        self.cuda_avail = cuda_avail

        self.epoch_ref = 0
        self.batch_norma = batch_norma
        #self.attention_layer = torch.nn.Sequential(
        #        torch.nn.Linear(self.hidden_dim*2,self.hidden_dim*2),
        #        torch.nn.ReLU(inplace=True)
        #        )
        if cuda_avail :
            self.device = torch.device("cuda")
        else:
            self.device = None

    def prepare_batch(self, x, y, phone):
        """
        :param x: list of B(batchsize) acoustic trajectories of variable lenghts,
        each element of the list is an array (K,18)  (K not always the same)
        :param y: list of B(batchsize) articulatory features,
        each element of the list is an array (K,429) (K not always the same)
        :return: 2 np array of sizes (B, K_max, 18) and (B, K_max, 429
        x,y initially data of the batch with different sizes . the script zeropad the acoustic and
        articulatory sequences so that all element in the batch have the same size
        """

        #max_length = np.max([len(phrase) for phrase in x])
        max_length = 300
        B = len(x)  # often batch size but not for validation
        new_x = torch.zeros((B, max_length, self.input_dim), dtype=torch.double)
        new_y = torch.zeros((B, max_length, 12), dtype=torch.double)
        new_phone = torch.zeros((B,max_length,61),dtype=torch.double)
        for j in range(B):
            zeropad = torch.nn.ZeroPad2d((0, 0, 0, max_length - len(x[j])))
            new_x[j] = zeropad(torch.from_numpy(x[j])).double()
            new_y[j] = zeropad(torch.from_numpy(y[j])).double()
            new_phone[j] = zeropad(torch.from_numpy(phone[j])).double()
        x = new_x.view((B, max_length, 40))
        y = new_y.view((B, max_length, 12))
        phone = new_phone.view((B,max_length,61))

        return x, y, phone
    #def attention_net_w(self, lstm_out, lstm_hidden):
    #    lstm_tmp_out = torch.chunk(lstm_out,2,-1)
    #    h = lstm_out[0]+lstm_out[1]
    #    lstm_hidden = torch.sum(lstm_hidden,dim = 1)
    #    lstm_hidden = lstm_hidden.unsqueeze(1)
    #    atten_w = self.attention_layer(lstm_hidden)
    #    m = torch.nn.Tanh()(h)
    #    atten_context = torch.bmm(atten_w,m.transpose(1,2))
    #    softmax_w = torch.nn.Functional.softmax(atten_context,dim = -1)
    #    context = torch.bmm(softmax_w,h).squeeze(1)
    #    return context
    def transform(self,acf):
        k = acf.size()
        #print(k)
        re = RandomErasing()
        for i in range(k[0]):
            acf[i] = re(acf[i])
        return acf

    def forward(self, x, filter_output=None):
        """
        :param x: (Batchsize,K,429)  acoustic features corresponding to batch size
        :param filter_output: whether or not to pass throught the convolutional layer
        :return: the articulatory prediction (Batchsize, K,18) based on the current weights
        """
        if filter_output is None :
            filter_output = (self.filter_type != "out")
        K = x.size()
        #print(K)
        m = K[1]
        #print(m)
        x = x.permute(0,2,1)
        mfcc = x[:,0:40,:]
        phone = x[:,40:101,:]
        

        #语音分解特征模块
        speaker_idt = x[:,101:357,:] #SDNfeature
        speaker_idt = speaker_idt.permute(0,2,1)
        speaker_idt = self.fc2(speaker_idt); #原始的个性化特征
        # print(speaker_idt.size())  #4*300*300

        #首先phone进行特征提取，得到预测的ema坐标
        # phone_out = self.phone_feature_extraction(phone);
        # # print(phone_out.size())  4*640*300
        # # 再将phone送入到lstm中，进行phone逆推
        # phone_out = phone_out.permute(0, 2, 1)  #4*300*640
        phone = phone.permute(0,2,1)
        y_all_fu = self.phone_extractor(phone)  #音素辅助特征
        # print(y_all_fu.size())  #4*300*12
        
        

        #局部特征
        # mfcc = mfcc.permute(0,2,1)
        acf = self.CNN_1D(mfcc)  # 4*640*300
        acf = acf.permute(0,2,1)
        #print(out.shape)

        #全局特征
        mfcc = mfcc.permute(0,2,1)
        out, *_ = self.encoder(mfcc);  
        # print(out.shape) #4*300*40

        
        # print(acf.size());
        # print(out.size())
        # print(phone.size())
        # print(speaker_idt.size())
        # print(y_all_fu.size())
        Input = torch.cat((mfcc,phone,speaker_idt,y_all_fu),2)  #4*300*413
        # print(Input.shape)
        # Input = self.fc1(Input)
        #print(Input.shape)
        y_fu = self.extractor(Input)
        Input_new = torch.cat((Input,y_fu),2)
        #acoustic_lip = torch.cat((x,lip),2)
        y_pred = self.generator(Input_new)
        #print(y_pred.shape)
        #print(y_fu.shape)
        return y_pred, y_fu, y_all_fu


    def get_filter_weights(self):
        """
        :return: low pass filter weights based on calculus exclusively using tensors so pytorch compatible
        """
        cutoff = torch.tensor(self.cutoff, dtype=torch.float64,requires_grad=True).view(1, 1)
        fc = torch.div(cutoff,
              self.sampling_rate)  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
        if fc > 0.5:
            raise Exception("cutoff frequency must be at least twice sampling rate")
        b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
        N = int(np.ceil((4 / b)))  # le window
        if not N % 2:
            N += 1  # Make sure that N is odd .
        self.N = N

        n = torch.arange(N).double()
        alpha = torch.mul(fc, 2 * (n - (N - 1) / 2)).double()
        minim = torch.tensor(0.01, dtype=torch.float64) #utile ?
        alpha = torch.max(alpha,minim)#utile ?
        h = torch.div(torch.sin(alpha), alpha)
        beta = n * 2 * math.pi / (N - 1)
        w = 0.5 * (1 - torch.cos(beta))  # Compute hanning window.
        h = torch.mul(h, w)  # Multiply sinc filter with window.
        h = torch.div(h, torch.sum(h))
        return h

    def get_filter_weights_en_dur(self):
        """
        :return:  low pass filter weights using classical primitive (not on tensor)
        """
        fc = self.cutoff / self.sampling_rate
        if fc > 0.5:
            raise Exception("La frequence de coupure doit etre au moins deux fois la frequence dechantillonnage")
        b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
        N = int(np.ceil((4 / b)))  # le window
        if not N % 2:
            N += 1  # Make sure that N is odd.
        self.N = N
        n = np.arange(N)
        h = np.sinc(fc * 2 * (n - (N - 1) / 2))
        w = 0.5 * (1 - np.cos(n * 2 * math.pi / (N - 1)))  # Compute hanning window.
        h = h * w
        h = h / np.sum(h)
        return torch.tensor(h)

    def init_filter_layer(self):
        """
        intialize the weights of the convolution in the NN,  so that it acts as a lowpass filter.
        the typefilter determines if the weights will be updated during the optim of the NN
        """


        # maybe the two functions do exactly the same...

        if self.filter_type in ["out","fix"] :
            weight_init = self.get_filter_weights_en_dur()
        elif self.filter_type == "unfix":
            weight_init = self.get_filter_weights()
        C_in = 1
        stride = 1
        must_be_5 = 5
        padding = int(0.5 * ((C_in - 1) * stride - C_in + must_be_5)) + 23
        weight_init = weight_init.view((1, 1, -1))
        lowpass = torch.nn.Conv1d(C_in, self.output_dim, self.N, stride=1, padding=padding, bias=False)

        if self.filter_type == "unfix":  # we let the weights move
            lowpass.weight = torch.nn.Parameter(weight_init,requires_grad=True)

        else :  # "out" we don't care the filter won't be applied, or "fix" the wieghts are fixed
            lowpass.weight = torch.nn.Parameter(weight_init,requires_grad=False)

        lowpass = lowpass.double()
        self.lowpass = lowpass

    def filter_layer(self, y):
        """
        :param y: (B,L,18) articulatory prediction not smoothed
        :return:  smoothed articulatory prediction
        apply the convolution to each articulation, maybe not the best solution (changing n_channel of the conv layer ?)
        """
        B = len(y)
        L = len(y[0])
        y = y.double()
        y_smoothed = torch.zeros(B, L, self.output_dim)
        for i in range(self.output_dim):
            traj_arti = y[:, :, i].view(B, 1, L)
            traj_arti_smoothed = self.lowpass(traj_arti)
            traj_arti_smoothed = traj_arti_smoothed.view(B, L)
            y_smoothed[:, :, i] = traj_arti_smoothed
        return y_smoothed

    def plot_results(self, y_target = None, y_pred_smoothed=None, y_pred_not_smoothed= None, to_cons=[]):
        """
        :param y: one TRUE arti trajectory
        :param y_pred_not_smoothed: one predicted arti trajectory not smoothed (forward with filtered=False)
        :param y_pred_smoothed:  one predicted arti trajectory (smoothed)
        :param to_cons:  articulations available to consider (list of 0/1)
        save the graph of each available trajectory predicted and true.
        If y_pred is given, also plot the non smoothed pred
        [future work : change filename and title of the graph]
        """
        print("you chose to plot")
        plt.figure()
        articulators = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
         'ul_x', 'ul_y', 'll_x', 'll_y', 'la','lp','ttcl','tbcl','v_x', 'v_y']
        idx_to_cons = [k for k in range(len(to_cons)) if to_cons[k]]
        for j in idx_to_cons:
            plt.figure()

            plt.plot(y_target[:, j])
            plt.plot(y_pred_smoothed[:, j])
            if y_pred_not_smoothed is not None:
                plt.plot(y_pred_not_smoothed[:, j], alpha=0.6)
            plt.title("{0}_{1}.png".format(self.name_file, articulators[j]))
            if y_pred_not_smoothed is not None:
                plt.legend(["target", "pred smoothed", "pred not smoothed"])
            else:
                plt.legend(["target", "pred smoothed"])
            save_pics_path = os.path.join(
                "images_predictions\\{0}_{1}.png".format(self.name_file, articulators[j]))
            plt.savefig(save_pics_path)
            plt.close('all')

    def evaluate_on_test(self, X_test, Y_test, phone, std_speaker, to_plot=False, to_consider=None, verbose=True, index_common = [], no_std = False):
        """
        :param X_test:  list of all the input of the test set
        :param Y_test:  list of all the target of the test set
        :param std_speaker : list of the std of each articulator, useful to calculate the RMSE of the predicction
        :param to_plot: wether or not we want to save some predicted smoothed and not and true trajectory
        :param to_consider: list of 0/1 for the test speaker , 1 if the articulator is ok for the test speaker
        :return: print and return the pearson correlation and RMSE between real and predicted trajectories per articulators.
        """
        idx_to_ignore = [i for i in range(len(to_consider)) if not(to_consider[i])]
        all_diff = np.zeros((1, self.output_dim))
        all_pearson = np.zeros((1, self.output_dim))
        if to_plot:
            indices_to_plot = np.random.choice(len(X_test), 2, replace=False)
        model2 = ac2art_model()
        model2= model2.double().cuda()
        #file_weights = "saved_models/F01_spec_loss_90_filter_fix_bn_False_32.pt"
        # file_weights = "saved_models/F01_indep_onfi_loss_90_filter_fix_bn_False_0.pt"
        # loaded_state = torch.load(file_weights,map_location=self.device)
        # model2.load_state_dict(loaded_state)
        file_weights = "saved_models/F01_indep_Haskins_loss_90_filter_fix_bn_False_0.pt"
        loaded_state = torch.load(file_weights,map_location=self.device)
        model2.load_state_dict(loaded_state)
        for i in range(len(X_test)):
            L = len(X_test[i])
            #L = len(phone[i])
            x_torch = torch.from_numpy(X_test[i]).view(1, L, 40).double()  #x (1,L,40)
            phone_torch = torch.from_numpy(phone[i]).view(1,L,61).double()
            y = Y_test[i].reshape((L, 12))
            y = y[:,0:6]
            if index_common != []:
                y = get_right_indexes(y, index_common, shape = 2)
            if self.cuda_avail:
                x_torch = x_torch.to(device=self.device)
                phone_torch = phone_torch.to(device=self.device)
            Input = torch.cat((x_torch,phone_torch),2)
            mu,log_sigma,emb,dec,_ = model2(Input)
            eps = log_sigma.new(*log_sigma.size()).normal_(0,1)
            speech_torch = (mu+torch.exp(log_sigma/2)*eps)
            emb = emb.unsqueeze(2)
            new_input = torch.cat([emb for i in range(0,L)],2)
            speech_torch = speech_torch.permute(0,2,1).cuda()
            new_input = new_input.permute(0,2,1).cuda()


            x_torch = x_torch.cuda()
            phone_torch = phone_torch.cuda()
            #print("x",x_torch.shape)
            #print("p",phone_torch.shape)
            #print("speech",speech_torch.shape)
            #print("new",new_input.shape)

            input = torch.cat((x_torch,phone_torch,speech_torch,new_input),2)
                #phone_torch = phone_torch.to(device = self.device)
            #Input = torch.cat((x_torch,phone_torch),2)
            #y_pred_not_smoothed = self(x_torch, False).double() #output y_pred (1,L,13)
            y_pred_smoothed, y_fu, y_all_fu = self(input,False)
            if self.cuda_avail:
                #y_pred_not_smoothed = y_pred_not_smoothed.cpu()
                y_pred_smoothed = y_pred_smoothed.cpu()
            #y_pred_not_smoothed = y_pred_not_smoothed.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,13)
            y_pred_smoothed = y_pred_smoothed.detach().numpy().reshape((L, 6))  # y_pred (L,6)
            #if to_plot:
            #    if i in indices_to_plot:
            #        self.plot_results(y_target = y, y_pred_smoothed = y_pred_smoothed,
            #                          y_pred_not_smoothed = y_pred_not_smoothed, to_cons = to_consider)
            rmse = np.sqrt(np.mean(np.square(y - y_pred_smoothed), axis=0))  # calculate rmse
            rmse = np.reshape(rmse, (1, 6))

            std_to_modify = std_speaker
            if index_common != [] and not no_std:
                std_to_modify = get_right_indexes(std_to_modify, index_common, shape=1)
            std_to_modify = torch.from_numpy(std_to_modify)
            std_to_modify = std_to_modify[0:6]
            std_to_modify = std_to_modify.detach().numpy()
            #############
            rmse = rmse*std_to_modify  # unormalize
            all_diff = np.concatenate((all_diff, rmse))
            pearson = [0]*self.output_dim
            for k in range(self.output_dim):
                pearson[k] = np.corrcoef(y[:, k].T, y_pred_smoothed[:, k].T)[0, 1]
            pearson = np.array(pearson).reshape((1, self.output_dim))
            all_pearson = np.concatenate((all_pearson, pearson))
        all_pearson = all_pearson[1:]
        #if index_common == []:
        #    all_pearson[:, idx_to_ignore] = 0
        all_diff = all_diff[1:]
        #if index_common == []:
        #    all_diff[:, idx_to_ignore] = 0
        all_pearson[np.isnan(all_pearson)] = 0

        pearson_per_arti_mean = np.mean(all_pearson, axis=0)
        rmse_per_arti_mean = np.mean(all_diff, axis=0)
        if verbose:
            print("rmse final : ", np.mean(rmse_per_arti_mean[rmse_per_arti_mean != 0]))
            print("rmse mean per arti : \n", rmse_per_arti_mean)
            print("pearson final : ", np.mean(pearson_per_arti_mean[pearson_per_arti_mean != 0]))
            print("pearson mean per arti : \n", pearson_per_arti_mean)

        return rmse_per_arti_mean, pearson_per_arti_mean

    def evaluate_on_test_modified(self,X_test, Y_test, std_speaker, to_plot=False, to_consider=None, verbose=True, index_common = [], no_std = False):
        """
        :param X_test:  list of all the input of the test set
        :param Y_test:  list of all the target of the test set
        :param std_speaker : list of the std of each articulator, useful to calculate the RMSE of the predicction
        :param to_plot: wether or not we want to save some predicted smoothed and not and true trajectory
        :param to_consider: list of 0/1 for the test speaker , 1 if the articulator is ok for the test speaker
        :return: print and return the pearson correlation and RMSE between real and predicted trajectories per articulators.
        """
        idx_to_ignore = [i for i in range(len(to_consider)) if not(to_consider[i])]
        all_diff = np.zeros((1, self.output_dim))
        all_diff_without_std = np.zeros((1, self.output_dim))
        all_pearson = np.zeros((1, self.output_dim))
        if to_plot:
            indices_to_plot = np.random.choice(len(X_test), 2, replace=False)
        for i in range(len(X_test)):
            L = len(X_test[i])
            x_torch = torch.from_numpy(X_test[i]).view(1, L, self.input_dim)  #x (1,L,429)
            y = Y_test[i].reshape((L, 18))                     #y (L,13)
            if index_common != []:
                y = get_right_indexes(y, index_common, shape = 2)
            if self.cuda_avail:
                x_torch = x_torch.to(device=self.device)
            y_pred_not_smoothed = self(x_torch, False).double() #output y_pred (1,L,13)
            y_pred_smoothed = self(x_torch, True).double()
            if self.cuda_avail:
                y_pred_not_smoothed = y_pred_not_smoothed.cpu()
                y_pred_smoothed = y_pred_smoothed.cpu()
            y_pred_not_smoothed = y_pred_not_smoothed.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,13)
            y_pred_smoothed = y_pred_smoothed.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,13)
            if to_plot:
                if i in indices_to_plot:
                    self.plot_results(y_target = y, y_pred_smoothed = y_pred_smoothed,
                                      y_pred_not_smoothed = y_pred_not_smoothed, to_cons = to_consider)
            rmse = np.sqrt(np.mean(np.square(y - y_pred_smoothed), axis=0))  # calculate rmse
            rmse = np.reshape(rmse, (1, self.output_dim))

            std_to_modify = std_speaker
            if index_common != [] and not no_std:
                std_to_modify = get_right_indexes(std_to_modify, index_common, shape=1)

            rmse_with_std = rmse*std_to_modify  # unormalize

            all_diff = np.concatenate((all_diff, rmse_with_std))
            all_diff_without_std = np.concatenate((all_diff_without_std, rmse))
            y_pred_smoothed = y_pred_smoothed.reshape(1,L, self.output_dim)
            y_pred_smoothed = torch.from_numpy(y_pred_smoothed)
            y = torch.from_numpy(y.reshape(1, L, self.output_dim))
            pearson_per_art = criterion_pearson_no_reduction(y, y_pred_smoothed, cuda_avail=self.cuda_avail, device=self.device) # (1,1,18)
            pearson_per_art = pearson_per_art.reshape(1, self.output_dim)
            all_pearson = np.concatenate((all_pearson, pearson_per_art))
        all_pearson = all_pearson[1:]
        if index_common == []:
            all_pearson[:, idx_to_ignore] = 0
        all_diff = all_diff[1:]
        if index_common == []:
            all_diff[:, idx_to_ignore] = 0
        all_pearson[np.isnan(all_pearson)] = 0

        pearson_per_arti_mean = np.mean(all_pearson, axis=0)
        rmse_per_arti_mean = np.mean(all_diff, axis=0)
        rmse_per_art_mean_without_std = np.mean(all_diff_without_std, axis=0)
        if verbose:
            print("rmse final : ", np.mean(rmse_per_arti_mean[rmse_per_arti_mean != 0]))
            print("rmse mean per arti : \n", rmse_per_arti_mean)
            print("pearson final : ", np.mean(pearson_per_arti_mean[pearson_per_arti_mean != 0]))
            print("pearson mean per arti : \n", pearson_per_arti_mean)

        return rmse_per_arti_mean, rmse_per_art_mean_without_std, pearson_per_arti_mean



