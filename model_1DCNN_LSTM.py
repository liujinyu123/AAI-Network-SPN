import torch
import torch.nn as nn
import os, sys, inspect
from torch.autograd import Variable

from Training.tools_learning import get_right_indexes

Currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(Currentdir)
sys.path.insert(0,parentdir)
import math
# import matplotlib.pyplot as plt
import numpy as np
import gc

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

class Gen(torch.nn.Module):
    def __init__(self,input_size=400,hidden_size=150,num_layer=3,output_size=6):
        super(Gen,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.output_size = output_size
        self.lstm_layer = nn.LSTM(input_size,hidden_size,num_layer,batch_first=True,bidirectional=True)
        self.layer2 = nn.Linear(hidden_size*2,output_size)
        self.LeakyReLU = nn.LeakyReLU(True)
    def forward(self,x):
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm_layer(x)
        out = self.layer2(lstm_output)
        return out

class Fusion(torch.nn.Module):
    def __init__(self):
        super(Fusion,self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(640,300),nn.LeakyReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(61,100),nn.LeakyReLU(True))
    def forward(self,x):
        x = x.permute(0,2,1)
        a = self.fc1(x[:,:,0:640])
        b = self.fc2(x[:,:,640:701])
        return torch.cat((a,b),2)

class Fus(torch.nn.Module):
    def __init__(self):
        super(Fus,self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(400,280),nn.LeakyReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(6,20),nn.LeakyReLU(True))
    def forward(self,x):
        a = self.fc1(x[:,:,0:400])
        b = self.fc2(x[:,:,400:406])
        return torch.cat((a,b),2)

class my_ac2art_model(torch.nn.Module):
    def __init__(self,hidden_dim, input_dim, output_dim, batch_size, name_file="",sampling_rate=100,
                  cutoff=10,cuda_avail =False, filter_type=1, batch_norma=False):
        super(my_ac2art_model,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        #self.conv1 = nn.Conv1d(in_channels=120,out_channels=128,kernel_size=1)
        #self.conv3 = nn.Conv1d(in_channels=120,out_channels=128,kernel_size=3,padding=1)
        #self.conv5 = nn.Conv1d(in_channels=120,out_channels=128,kernel_size=5,padding=2)
        #self.conv7 = nn.Conv1d(in_channels=120,out_channels=128,kernel_size=7,padding=3)
        #self.conv9 = nn.Conv1d(in_channels=120,out_channels=128,kernel_size=9,padding=4)
        #self.pool1 = nn.MaxPool1d(1)
        #self.pool3 = nn.MaxPool1d(3,stride=1,padding=1)
        #self.pool5 = nn.MaxPool1d(5,stride=1,padding=2)
        #self.pool7 = nn.MaxPool1d(7,stride=1,padding=3)
        #self.pool9 = nn.MaxPool1d(9,stride=1,padding=4)
        #self.LeakyReLU = nn.LeakyReLU()
        self.CNN_1D = CNN_1D()
        self.Fusion = Fusion()
        self.Fus = Fus()
        self.Gen = Gen(input_size=400)
        self.Gen_t = Gen(input_size=300)
        #self.Linear = nn.Linear(600,400)
        #self.Linear2 = nn.Linear(200,400)
        #self.Linearfinal = nn.Linear(1080,540)
        #self.Linearfinal1 = nn.Linear(540,270)
        #self.Linearfinal2 = nn.Linear(600,300)
        #self.lstm_layer1 = nn.LSTM(input_size=331,hidden_size=150,num_layers=1,
        #        bidirectional=True)
        #self.batch_norm_layer = nn.BatchNorm1d(hidden_dim*2)
        #self.lstm_layer2 = nn.LSTM(input_size=300,hidden_size=150,num_layers=1,
        #        bidirectional=True)
        #self.lstm_layer3 = nn.LSTM(input_size=300,hidden_size=150,num_layers=1,
        #        bidirectional=True)
       
        #self.readout_layer = nn.Linear(300,output_dim)
        self.epoch_ref = 0
        self.sampling_rate = sampling_rate
        self.cutoff = cutoff
        self.N = None
        self.min_valid_error = 100000
        self.all_training_loss = []
        self.all_validation_loss = []
        self.all_test_loss = []
        self.name_file = name_file
        self.lowpass = None
        self.cuda_avail = True

        if cuda_avail:
            self.device = torch.device("cuda")
        else:
            self.device = None

    def prepare_batch(self,x,y,phone):
        max_length = np.max([len(phrase) for phrase in x])
        B = len(x)  # often batch size but not for validation
        new_x = torch.zeros((B, max_length, self.input_dim), dtype=torch.double)
        new_y = torch.zeros((B, max_length, 12), dtype=torch.double)
        new_phone = torch.zeros((B,max_length,61),dtype=torch.double)
        for j in range(B):
            zeropad = torch.nn.ZeroPad2d((0, 0, 0, max_length - len(x[j])))
            new_x[j] = zeropad(torch.from_numpy(x[j])).double()
            new_y[j] = zeropad(torch.from_numpy(y[j])).double()
            new_phone[j] = zeropad(torch.from_numpy(phone[j])).double()
        x = new_x.view((B, max_length, self.input_dim))
        y = new_y.view((B, max_length, 12))
        phone = new_phone.view((B,max_length,61))

        return x, y, phone
    def forward(self,x):
        x = x.permute(0,2,1)
        acf = x[:,0:40,:]
        phone = x[:,40:101,:]
        out = self.CNN_1D(acf)
        Input = self.Fusion(torch.cat((out,phone),1))
        # acf = acf.permute(0,2,1)
        y_fu = self.Gen(Input)
        # print(y_fu.shape)
        #y_pred = self.Gen_t(self.Fus(torch.cat((Input,y_fu),2)))
        return y_fu




    """
    def forward(self,x):
        ## :Extract features using 1DCNN
        x = x.permute(0,2,1)
        #print(x.shape)
        acf = x[:,0:120,:]
        phone = x[:,120:181,:]
        #print("acf",acf.shape)
        #print("phone",phone.shape)
        out1 = acf
        out2 = acf
        out3 = acf
        for i in range(3):
            #print("out1",out1.shape)
            #print("out2",out2.shape)
            #print("out3",out3.shape)
            out1 = self.pool1(self.LeakyReLU(self.conv1(out1)))
            out2 = self.pool3(self.LeakyReLU(self.conv3(out2)))
            out3 = self.pool5(self.LeakyReLU(self.conv5(out3)))
            #print("out1",out1.shape)
            #print("out2",out2.shape)
            #print("out3",out3.shape)
            if i == 0:
                output1 = out1
                output2 = out2
                output3 = out3
            else:
                output1 = torch.cat((output1,out1),1)
                output2 = torch.cat((output2,out2),1)
                output3 = torch.cat((output3,out3),1)
        out = torch.cat((output1,output2,output3),1)
        out = out.permute(0,2,1)
        out = self.Linearfinal(out)
        out = self.Linearfinal1(out)
        #out = self.Linearfinal2(out)
        #print("out",out.shape)
        phone = phone.permute(0,2,1)
        #print("phone",phone.shape)
        out = torch.cat((out,phone),2)
        ## : AAI model

        #print("01",out.shape)
        lstm_out, hidden_dim = self.lstm_layer1(out)
        #B = lstm_out.shape[0] #presque tjrs batch size
        #if self.batch_norma :
        #lstm_out_temp = lstm_out.view(B,2*self.hidden_dim,-1)
        #lstm_out_temp = torch.nn.functional.relu(self.batch_norm_layer(lstm_out_temp))
        #lstm_out = lstm_out_temp.view(B,  -1,2 * self.hidden_dim)
        lstm_out = torch.nn.functional.relu(lstm_out)
        
        #print("12",lstm_out.shape)

        lstm_out, hidden_dim = self.lstm_layer2(lstm_out)
        lstm_out=torch.nn.functional.relu(lstm_out)

        #print("23",lstm_out.shape)

        lstm_out, hidden_dim = self.lstm_layer3(lstm_out)
        lstm_out=torch.nn.functional.relu(lstm_out)
        y_pred = self.readout_layer(lstm_out)
        #print(y_pred.shape)
        return y_pred
    """
    def evaluate_on_test(self,X_test,Y_test,phone,std_speaker,to_plot=False, to_consider=None, verbose=True, index_common = [], no_std = False):
        idx_to_ignore = [i for i in range(len(to_consider)) if not(to_consider[i])]
        all_diff = np.zeros((1, self.output_dim))
        all_pearson = np.zeros((1, self.output_dim))
        if to_plot:
            indices_to_plot = np.random.choice(len(X_test), 2, replace=False)
        for i in range(len(X_test)):
            #L = len(X_test[i])
            L = len(phone[i])
            x_torch = torch.from_numpy(X_test[i]).view(1, L, 40)  #x (1,L,400)
            phone_torch = torch.from_numpy(phone[i]).view(1,L,61)
            #y = Y_test[i].reshape((L, 12))#y (L,13)
            y = torch.from_numpy(Y_test[i]).view(L,12)
            y = y[:,0:6]
            y = y.detach().numpy().reshape((L,6))
            #if index_common != []:
            #    y = get_right_indexes(y, index_common, shape = 2)
            if self.cuda_avail:
                x_torch = x_torch.to(device=self.device)
                phone_torch = phone_torch.to(device = self.device)
            x_torch = x_torch.float()
            phone_torch = phone_torch.float()
            Input = torch.cat((x_torch,phone_torch),2)
            Input = Input.double()
            y_pred_smoothed = self(Input) #output y_pred (1,L,6)
            #y_pred_smoothed = self(, True).double()
            if self.cuda_avail:
                #y_pred_not_smoothed = y_pred_not_smoothed.cpu()
                y_pred_smoothed = y_pred_smoothed.cpu()
            #y_pred_not_smoothed = y_pred_not_smoothed.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,6)
            y_pred_smoothed = y_pred_smoothed.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,6))
            # if to_plot:
            #     if i in indices_to_plot:
            #         self.plot_results(y_target = y, y_pred_smoothed = y_pred_smoothed,
            #                           y_pred_not_smoothed = y_pred_not_smoothed, to_cons = to_consider)
            rmse = np.sqrt(np.mean(np.square(y - y_pred_smoothed), axis=0))  # calculate rmse
            rmse = np.reshape(rmse, (1, self.output_dim))

            std_to_modify = std_speaker
            if index_common != [] and not no_std:
                std_to_modify = get_right_indexes(std_to_modify, index_common, shape=1)
            #print(std_to_modify)
            #print(std_to_modify.size())
            std_to_modify = torch.from_numpy(std_to_modify)
            #print(std_to_modify.shape)
            std_to_modify = std_to_modify[0:6]
            std_to_modify = std_to_modify.detach().numpy()
            #print("rmse",rmse.size())
            #print("std",std_to_modify.size())
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
             
