#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    Script to train the model for the articulatory inversion.
    Some parameters concern the model itself, others concern the data used.
    Creates a model with the class myac2artmodel for the asked parameters.
    Learn category by category (in a category  all the speakers have the same arti traj available), the gradients are put
    at 0 for the unavailable arti traj so that it learns only on correct data.
    The model stops training by earlystopping if the validation score is several time consecutively increasing
    The weights of the model are saved in Training/saved_models/name_file.txt, with name file containing the info
    about the training/testing set [and not about the model parameters].

    The results of the model are evaluated on the test set, and are the averaged rmse and pearson per articulator.
    Those results are added as 2 new lines in the file "model_results.csv" , with 1 column being the name of the model
    and the last column the number of epochs [future work : add 1 columns per argument to store ALL the info about
    the model]


"""
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


import os
"""
ncpu="10" # number of cpu available
os.environ["OMP_NUM_THREADS"] = ncpu  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = ncpu  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = ncpu  # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = ncpu  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = ncpu  # export NUMEXPR_NUM_THREADS=4"""
import numpy as np
import argparse
from Training.model_learning_F01_phone import my_ac2art_model
from Training.model_learning import ac2art_model
import torch
import os
import csv
from Training.pytorchtools import EarlyStopping
import random
from Training.tools_learning import which_speakers_to_train_on, give_me_train_valid_test_filenames, \
    cpuStats, memReport, criterion_both_phone, load_np_ema_and_mfcc_and_phone, plot_filtre, criterion_pearson
import json

root_folder = os.path.dirname(os.getcwd())

def train_model(test_on, n_epochs, loss_train, patience, select_arti, corpus_to_train_on, batch_norma, filter_type,
                to_plot, lr, delta_test, config, speakers_to_train_on = "", speakers_to_valid_on = "", relearn = False):
    """
    :param test_on: (str) one speaker's name we want to test on, the speakers and the corpus the come frome can be seen in
    "fonction_utiles.py", in the function "get_speakers_per_corpus'.

    :param n_epochs: (int)  max number of epochs for the training. We use an early stopping criterion to stop the training,
    so usually we dont go through the n_epochs and the early stopping happends before the 30th epoch (1 epoch is when
    have trained over ALL the data in the training set)

    :param loss_train: (int) alpha in the combined loss . can be anything between 0 and 100.
    the loss is the combinated loss alpha*rmse/1000+(1-alpha)*pearson.

    :param patience: (int) the number successive epochs with a validation loss increasing before stopping the training.
    We usually set it to 5. The more data we have, the smaller it can be (i think)

    :param select_arti: (bool) always true, either to use the trick to only train on available articulatory trajectories,
    fixing the predicted trajectory (to zero) and then the gradient will be 0.

    :param corpus_to_train_on: (list) list of the corpuses to train on. Usually at least the corpus the testspeaker comes from.
    (the testspeaker will be by default removed from the training speakers).

    :param batch_norma: (bool) whether or not add batch norm layer after the lstm layers (maybe better to add them after the
    feedforward layers? )

    :param filter_type: (int) either 0 1 or 2. 0 the filter is outside of the network, 1 it is inside and the weight are fixed
    during the training, 2 the weights get adjusted during the training

    :param to_plot: (bool) if true the trajectories of one random test sentence are saved in "images_predictions"

    :param lr: initial learning rate, usually 0.001

    :param delta_test: frequency of validation evaluation, 1 seems good

    :param config : either "spe" "dep", or "indep", for specific (train only on test sp), dependant (train on test sp
    and others), or independant, train only on other speakers

    :return: [rmse, pearson] . rmse the is the list of the 18 rmse (1 per articulator), same for pearson.
    """
    f_loss_train = open('training_loss.csv', 'w')
    f_loss_valid = open('valid_loss.csv', 'w')
    corpus_to_train_on = corpus_to_train_on[1:-1].split(",")
    speakers_to_train_on = speakers_to_train_on[1:-1].replace("'", "").replace('"', '').replace(' ', '').split(",")
    if speakers_to_train_on == [""] or speakers_to_train_on == []:
        train_on = which_speakers_to_train_on(corpus_to_train_on, test_on, config)
    else:
        train_on = speakers_to_train_on

    speakers_to_valid_on = speakers_to_valid_on[1:-1].replace("'", "").replace('"', '').replace(' ', '').split(",")
    if speakers_to_valid_on == [""] or speakers_to_valid_on == []:
        valid_on = []
    else:
        valid_on = speakers_to_valid_on
    print('train', train_on)
    print('valid', valid_on)
    print('test', test_on)
    name_corpus_concat = ""
    if config != "spec" : # if spec DOESNT train on other speakers
        for corpus in corpus_to_train_on:
            name_corpus_concat = name_corpus_concat + corpus + "_"

    name_file = test_on+"_"+config+"_"+name_corpus_concat+"loss_"+str(loss_train)+"_filter_"+\
                str(filter_type)+"_bn_"+str(batch_norma)

    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")

    previous_models = os.listdir("saved_models")
    previous_models_2 = [x[:len(name_file)] for x in previous_models if x.endswith(".txt")]
    n_previous_same = previous_models_2.count(name_file)  # how many times our model was trained

    if n_previous_same > 0:
        print("this models has alread be trained {} times".format(n_previous_same))
    else :
        print("first time for this model")
    name_file = name_file + "_" + str(n_previous_same)  # each model trained only once ,
    # this script doesnt continue a previous training if it was ended ie if there is a .txt
    print("going to train the model with name",name_file)

    cuda_avail = torch.cuda.is_available()
    print(" cuda ?", cuda_avail)
    if cuda_avail:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hidden_dim = 150
    input_dim = 40
    batch_size = 4
    output_dim = 6
    early_stopping = EarlyStopping(name_file, patience=patience, verbose=True)
    model = my_ac2art_model(hidden_dim=hidden_dim, input_dim=input_dim, name_file=name_file, output_dim=output_dim,
                            batch_size=batch_size, cuda_avail=cuda_avail,
                            filter_type=filter_type, batch_norma=batch_norma)
    model = model.double()#逆推模型
    model1 = ac2art_model()#分离模型
    model1 = model1.double().cuda()
    #file_weights = os.path.join("saved_models", name_file +".pt")
    #file_weights = "saved_models/F01_indep_onif_loss_90_filter_fix_bn_False_1.pt"
    #file_weights = "saved_models/F01_spec_loss_90_filter_fix_bn_False_32.pt"
    file_weights = "saved_models/F01_indep_onfi_loss_90_filter_fix_bn_False_0.pt"
    loaded_state = torch.load(file_weights,map_location=device)
    model1.load_state_dict(loaded_state)

    if cuda_avail:
        model = model.to(device=device)
    relearn = False
    if relearn:
        print("&&&&&&&&&&")
        load_old_model = True
        if load_old_model:
            if os.path.exists(file_weights):
                print("previous model did not finish learning")
                loaded_state = torch.load(file_weights,map_location=device)
                model.load_state_dict(loaded_state)
                model_dict = model.state_dict()
                loaded_state = {k: v for k, v in loaded_state.items() if
                                k in model_dict}  # only layers param that are in our current model
                loaded_state = {k: v for k, v in loaded_state.items() if
                                loaded_state[k].shape == model_dict[k].shape}  # only if layers have correct shapes
                model_dict.update(loaded_state)
                model.load_state_dict(model_dict)



    files_per_categ, files_for_test = give_me_train_valid_test_filenames(train_on=train_on,test_on=test_on,config=config,batch_size= batch_size, valid_on=valid_on)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    categs_to_consider = files_per_categ.keys()
    with open('categ_of_speakers.json', 'r') as fp:
        categ_of_speakers = json.load(fp)  # dict that gives for each category the speakers in it and the available arti
    plot_filtre_chaque_epochs = False
    print("音素模块的权重值为：")
    print(0.1)
    for epoch in range(n_epochs):
        weights = model.lowpass.weight.data[0, 0, :].cpu()
        if plot_filtre_chaque_epochs :
            plot_filtre(weights)
        n_this_epoch = 0
        random.shuffle(list(categs_to_consider))
        loss_train_this_epoch = 0
        loss_fu = 0
        loss_zhen = 0
        loss_phone = 0
        for categ in categs_to_consider:
            files_this_categ_courant = files_per_categ[categ]["train"]
            random.shuffle(files_this_categ_courant)
            while len(files_this_categ_courant) > 0: # go through all  the files batch by batch
                n_this_epoch+=1
                x, y,phone  = load_np_ema_and_mfcc_and_phone(files_this_categ_courant[:batch_size])

                files_this_categ_courant = files_this_categ_courant[batch_size:] #we a re going to train on this 10 files
                x,y,phone  = model.prepare_batch(x,y,phone) 
                if cuda_avail:
                   x,y,phone = x.to(device=model1.device).double(), y.to(device=model1.device).double(), phone.to(device=model1.device).double()
                #Input = torch.cat((x,phone),2)
                #print(phone.shape)
                mu,log_sigma,emb,dec = model1(x)
                eps = log_sigma.new(*log_sigma.size()).normal_(0,1)
                speech_torch = (mu+torch.exp(log_sigma/2)*eps)
                emb = emb.unsqueeze(2)
                new_input = torch.cat([emb for i in range(0,300)],2)
                speech_torch = speech_torch.permute(0,2,1).cuda()
                new_input = new_input.permute(0,2,1).cuda()
                input = torch.cat((x,phone,speech_torch,new_input),2)
                y_pred, y_fu, y_all_fu = model(input)
                if cuda_avail:
                    y_pred = y_pred.to(device=device)
                    y_fu = y_fu.to(device=device)
                    y_all_fu = y_all_fu.to(device=device)
                y1 = y[:,:,0:6]
                y2 = y[:,:,6:12]
                optimizer.zero_grad()
                if select_arti:
                    arti_to_consider = categ_of_speakers[categ]["arti"]  # liste de 18 0/1 qui indique les arti à considérer
                    idx_to_ignore = [i for i, n in enumerate(arti_to_consider) if n == "0"]
                    y_pred[:, :, idx_to_ignore] = 0 #the grad associated to this value will be zero  : CHECK THAT
                    # y_pred[:,:,idx_to_ignore].detach()
                    #y[:,:,idx_to_ignore].requires_grad = False

                loss = criterion_both_phone(y1, y_pred, y2, y_fu, y, y_all_fu, cuda_avail = cuda_avail, device=device)
                loss.backward()
                optimizer.step()

                # computation to have evolution of the losses
                torch.cuda.empty_cache()
                loss_2 = torch.nn.MSELoss(reduction='mean')(y2, y_fu)
                loss_fu += loss_2.item()
                loss_3 = torch.nn.MSELoss(reduction='mean')(y1, y_pred)
                loss_zhen += loss_3.item()
                loss_4 = torch.nn.MSELoss(reduction='mean')(y, y_all_fu)
                loss_phone += loss_4.item()
                loss_train_this_epoch += loss.item()
        #rmse = get_test_rmse_seq(model)
        torch.cuda.empty_cache()

        loss_train_this_epoch = loss_train_this_epoch/n_this_epoch
        loss_fu_this_epoch = loss_fu/n_this_epoch
        loss_zhen_this_epoch = loss_zhen/n_this_epoch
        loss_phone_this_epoch = loss_phone/n_this_epoch
        print("Training loss for epoch", epoch, ': ', loss_train_this_epoch)
        print("y_fu loss", epoch, ": ",loss_fu_this_epoch)
        print("y_zhen loss", epoch, ": ",loss_zhen_this_epoch)
        print("y_phone loss", epoch, ": ", loss_phone_this_epoch)
        if epoch >= 0 :
            random.shuffle(files_for_test)
            x, y, phone = load_np_ema_and_mfcc_and_phone(files_for_test)
            print("evaluation on speaker {}".format(test_on))
            std_speaker = np.load(os.path.join(root_folder,"Preprocessing","norm_values","std_ema_"+test_on+".npy"))
            arti_per_speaker = os.path.join(root_folder, "Preprocessing", "articulators_per_speaker.csv")
            csv.register_dialect('myDialect', delimiter=';')
            with open(arti_per_speaker, 'r') as csvFile:
                reader = csv.reader(csvFile, dialect="myDialect")
                next(reader)
                for row in reader:
                    if row[0] == test_on:
                        arti_to_consider = row[1:19]
                        arti_to_consider = [int(x) for x in arti_to_consider]

            rmse_per_arti_mean, pearson_per_arti_mean = model.evaluate_on_test(x, y, phone,  std_speaker = std_speaker, to_plot=to_plot
                                                                       , to_consider = arti_to_consider)
        torch.save(model.state_dict(), os.path.join("saved_models",model.name_file+".pt"))
        model.epoch_ref = model.epoch_ref + epoch  # voir si ca marche vrmt pour les rares cas ou on continue un training
        model.load_state_dict(torch.load(os.path.join("saved_models",name_file+'.pt')))
        torch.save(model.state_dict(), os.path.join( "saved_models",name_file+".txt")) #lorsque .txt ==> training terminé !
                   
     
    
    random.shuffle(files_for_test)
    x, y, phone = load_np_ema_and_mfcc_and_phone(files_for_test)
    print("evaluation on speaker {}".format(test_on))
    std_speaker = np.load(os.path.join(root_folder,"Preprocessing","norm_values","std_ema_"+test_on+".npy"))
    arti_per_speaker = os.path.join(root_folder, "Preprocessing", "articulators_per_speaker.csv")
    csv.register_dialect('myDialect', delimiter=';')
    with open(arti_per_speaker, 'r') as csvFile:
        reader = csv.reader(csvFile, dialect="myDialect")
        next(reader)
        for row in reader:
            if row[0] == test_on:
                arti_to_consider = row[1:19]
                arti_to_consider = [int(x) for x in arti_to_consider]
    rmse_per_arti_mean, pearson_per_arti_mean = model.evaluate_on_test(x, y,phone, std_speaker = std_speaker, to_plot=to_plot
                                                                       , to_consider = arti_to_consider)



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Train and save a model.')

    parser.add_argument('test_on', type=str,
                        help='the speaker we want to test on')

    parser.add_argument('--speakers_to_train', type=str, default="",
                        help='specific speakers to train on')

    parser.add_argument('--speakers_to_valid', type=str, default="",
                        help='specific speakers to valid on')

    parser.add_argument('--n_epochs', type=int, default=20,
                        help='max number of epochs to train the model')

    parser.add_argument("--loss_train",type = int, default=90,
                        help = "from 0 to 100, coeff of pearson is the combined loss")

    parser.add_argument("--patience",type=int, default=5,
                        help = "patience before early topping")

    parser.add_argument("--select_arti", type = bool,default=False,
                        help = "whether to learn only on available parameters or not")
    parser.add_argument('corpus_to_train_on', type=str,
                        help='list of the corpus we want to train on ')

    parser.add_argument('--batch_norma', type=bool, default= False,
                        help='whether to add batch norma after lstm layyers')

    parser.add_argument('--filter_type', type=str, default="out",
                        help='"out" filter outside of nn, "fix" filter with fixed weights, "unfix" filter with adaptable weights')

    parser.add_argument('--to_plot', type=bool, default= False,
                        help='whether to save one graph of prediction & target of the test ')

    parser.add_argument('--relearn', type=bool, default=False,
                        help='whether to learn on previous partially learned model or not')

    parser.add_argument('--lr', type = float, default = 0.001,
                        help='learning rate of Adam optimizer ')

    parser.add_argument('--delta_test', type=int, default=3,
                        help='how often evaluate the validation set')

    parser.add_argument('config', type=str,
                        help='spec or dep or train_indep or indep that stands for speaker specific/dependant/independant')

    args = parser.parse_args()
    print('arguments given:', args.test_on, args.speakers_to_train, args.n_epochs, args.loss_train,
          args.patience, args.select_arti, args.corpus_to_train_on, args.batch_norma, args.filter_type, args.to_plot,args.lr, args.delta_test, args.config )
    train_model(test_on=args.test_on, n_epochs=args.n_epochs, loss_train=args.loss_train,
                patience=args.patience, select_arti=args.select_arti, corpus_to_train_on=args.corpus_to_train_on,
                batch_norma=args.batch_norma, filter_type=args.filter_type, to_plot=args.to_plot,
                lr=args.lr, delta_test=args.delta_test, config=args.config, speakers_to_train_on=args.speakers_to_train,
                relearn=args.relearn, speakers_to_valid_on=args.speakers_to_valid)
