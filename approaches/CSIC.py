import sys, time, os
import numpy as np
import random
import torch
from copy import deepcopy
import utils
from utils import *
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import *
import math

sys.path.append('..')
from arguments import get_args

args = get_args()

from transformers import get_linear_schedule_with_warmup,AdamW
from CSIC_model.model import BERTCombine

import datetime
class Appr(object):
    

    def __init__(self, model, nepochs=100, sbatch=256, clipgrad=100, args=None, log_name=None, split=False,task_names = None):

        self.model = model
        self.original_network = deepcopy(self.model)
        file_name = log_name
        if not os.path.exists('result_data/csvdata/'):
            os.makedirs('result_data/csvdata/')
        self.logger = utils.logger(file_name=file_name, resume=False, path='result_data/csvdata/', data_format='csv')

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.clipgrad = clipgrad
        self.args = args
        self.iteration = 0
        self.split = split
        
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.tasks = task_names
        
        
        

        

        return



    def train(self, t, train_dataloader, test_dataloader, data,combine,epoch):
        if combine == True:
            self.finetuned_network = deepcopy(self.model)
            self.finetuned_network.eval()
            print("create combine model")
            self.model = BERTCombine(self.original_network,self.finetuned_network,alpha_init= 1/2)
            self.model.bert = torch.nn.DataParallel(self.model.bert)
            self.model = self.model.cuda()
            utils.freeze_model(self.original_network)
            utils.freeze_model(self.finetuned_network)
            self.model.set_dataset(self.tasks[t]) #set classifier layer
            no_decay = []
            alpha_beta = []
            for n,param in self.model.bert.module.named_parameters():
                if 'bias' in n or 'LayerNorm.weight' in n: #BERT not use weight_decay in bias and LN
                    no_decay.append(id(param))
                if 'alpha' in n or 'beta' in n:
                    # print(n)
                    alpha_beta.append(id(param))
            alpha_beta_param = filter(lambda p: p.requires_grad and id(p) in alpha_beta,self.model.parameters())
            params = [
                {"params":self.model.classifier.weight,"lr":5e-5,'weight_decay':1e-8},
                {"params":self.model.classifier.bias,"lr":5e-5,},
                {"params":alpha_beta_param,"lr":1e-3}      
            ]
            self.optimizer = AdamW(params,
                        eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
                        )

            # Total number of training steps is number of batches * number of epochs.
            total_steps = len(train_dataloader) * epoch
            print(len(train_dataloader))
            # Create the learning rate scheduler.
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                        num_warmup_steps = 0, # Default value in run_glue.py
                                                        num_training_steps = total_steps)
        else :
            self.original_network = deepcopy(self.model)
            self.original_network.eval()
            no_decay = []
            for n,param in self.model.bert.module.named_parameters():
                if 'bias' in n or 'LayerNorm.weight' in n: #BERT not use weight_decay in bias and LN
                    no_decay.append(id(param))
            bert_param = filter(lambda p: p.requires_grad and id(p) not in no_decay ,self.model.bert.module.parameters())
            bert_param_nodecay = filter(lambda p: p.requires_grad and id(p) and id(p) in no_decay,self.model.bert.module.parameters())
            params = [
                {"params":bert_param,"lr":5e-5,'weight_decay' : 1e-8},
                {"params":bert_param_nodecay,"lr":5e-5,'weight_decay' : 0},
                {"params":self.model.classifier.weight,"lr":5e-5,'weight_decay':1e-8},
                {"params":self.model.classifier.bias,"lr":5e-5,},      
            ]
            self.optimizer = AdamW(params,
                        eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
                        )

            # Total number of training steps is number of batches * number of epochs.
            total_steps = len(train_dataloader) * epoch
            print(len(train_dataloader))
            # Create the learning rate scheduler.
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                        num_warmup_steps = 0, # Default value in run_glue.py
                                                        num_training_steps = total_steps)
        # initial best_avg

        valid_acc_t = {}
        valid_acc_t_norm = {}

        # Loop epochs
        for e in range(epoch):
            

            # Train
            clock0 = time.time()
            avg = 0
            # num_batch = xtrain.size(0)
            num_batch = len(train_dataloader)
            self.model.set_dataset(self.tasks[t])
            self.train_epoch(t, train_dataloader,combine)
            
            clock1 = time.time()
            train_loss, train_acc = self.eval(t, train_dataloader,combine)
            
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1, 1000 * self.sbatch * (clock1 - clock0) / num_batch,
                1000 * self.sbatch * (clock2 - clock1) / num_batch, train_loss, 100 * train_acc), end='')
            # Valid
            
            valid_loss, valid_acc = self.eval(t, test_dataloader,combine)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')




            print()

        torch.save(self.model,'_task_{}.pt'.format(t))

        return
    def flat_accuracy(self,preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)



    def format_time(self,elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train_epoch(self,t,train_dataloader,combine,T = 2):
        device = 'cuda'
        total_loss = 0
        self.model.train()
        for step, batch in enumerate(tqdm(train_dataloader,desc="train")):
        # for step,batch in enumerate(train_dataloader):
            

            b_input_ids = batch[0].cuda()
            b_input_mask = batch[1].cuda()
            b_labels = batch[2].cuda()

            self.model.zero_grad()
            outputs = self.model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]


            if combine == True:
                
                original_outputs = self.original_network.bert.module(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,)
                finetuned_outputs = self.finetuned_network.bert.module(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,)
                #structure combined network
                structure_combined_outputs = self.model.bert.module(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,)
                loss_LWF = 0
                for i in range(t):
                    self.original_network.set_dataset(self.tasks[i])
                    self.model.set_dataset(self.tasks[i])
                    soft_target_old = self.original_network.classifier(original_outputs[1])
                    combine_outs = self.model.classifier(structure_combined_outputs[1])
                    outputs_O = F.softmax(soft_target_old.view(-1,2)/T,dim=1)
                    outputs_C = F.softmax(combine_outs.view(-1,2)/T,dim=1)
                    loss_LWF_t = outputs_O.mul(-1*torch.log(outputs_C))
                    loss_LWF_t = loss_LWF_t.sum(1)
                    loss_LWF += loss_LWF_t.mean()*T*T
                

                self.finetuned_network.set_dataset(self.tasks[t])
                self.model.set_dataset(self.tasks[t])
                soft_target_new = self.finetuned_network.classifier(finetuned_outputs[1])
                combine_outs = self.model.classifier(structure_combined_outputs[1])
                outputs_N = F.softmax(soft_target_new.view(-1,2)/T,dim=1)
                outputs_T = F.softmax(combine_outs.view(-1,2)/T,dim=1)
                loss_KD = outputs_N.mul(-1*torch.log(outputs_T))
                loss_KD = loss_KD.sum(1)
                loss_KD = loss_KD.mean()*T*T
                
                loss = loss_LWF/t + self.lambda_1* loss + (1-self.lambda_1) * loss_KD

            l1_loss = 0
            for n,p in self.model.named_parameters():
                if 'alpha' in n or 'beta' in n:
                    l1_loss += (self.lambda_2) * torch.norm(p,p=1)
            loss += l1_loss
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)            
            

        return


    def eval(self,t,test_dataloader,regular):

        t0 = time.time()
        device = 'cuda'

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in test_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():        
                # if regular:
                outputs = self.model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            loss = outputs[0]
            logits = outputs[1]

            eval_loss += loss.item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1
        # print(outputs[0].size())
        # Report the final accuracy for this validation run.
        avg_eval_loss = eval_loss/nb_eval_steps

        return avg_eval_loss, eval_accuracy/nb_eval_steps



   