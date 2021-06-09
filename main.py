import sys
import GPUtil
import os
from arguments import get_args
args = get_args()
GPU_id = GPUtil.getAvailable(maxLoad=0.05, maxMemory=0.05, limit=args.n_gpus)
gpus = ""
for i in GPU_id:
    gpus += str(i) + ","
os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

import torch
import numpy as np
import random

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


from approaches import CSIC as CSIC
from CSIC_model.model import BertClassifier_base
from dataset import bert_train_loader

from transformers import get_linear_schedule_with_warmup,AdamW
import torch.nn as nn
from copy import deepcopy



# print(gpus)


epochs = 3
epochs_combine = 5
lr = 5e-5
batch_size = 16
regular = False
model = BertClassifier_base()



test_loader=[]

tasks = ['magazines.task','apparel.task','health_personal_care.task','camera_photo.task','toys_games.task','software.task','baby.task','kitchen_housewares.task','sports_outdoors.task',
    'electronics.task','books.task','video.task','imdb.task','dvd.task','music.task','MR.task']
print(tasks)
CSIC_train = CSIC.Appr(model,epochs,batch_size,args = args,log_name=args.logname,task_names = tasks)
CSIC_train.model.bert = torch.nn.DataParallel(CSIC_train.model.bert,device_ids=[i for i in range(len(GPU_id))])

for t in range(0,16):
    train_dataloader,valid_dataloader,test_dataloader = bert_train_loader(tasks[t])
    test_loader.append(test_dataloader)
    CSIC_train.model.add_dataset(tasks[t],2)
    CSIC_train.model.set_dataset(dataset = tasks[t])
    CSIC_train.model = CSIC_train.model.cuda()


    print('training {}'.format(tasks[t]))
    if t==0:
        CSIC_train.train(t,train_dataloader,valid_dataloader,test_loader,combine = False,epoch = epochs+2)
    else:
        CSIC_train.train(t,train_dataloader,valid_dataloader,test_loader,combine = False,epoch = epochs)
        # train middle network
        CSIC_train.train(t,train_dataloader,valid_dataloader,test_loader,combine = True,epoch = epochs_combine)
    CSIC_train.model.set_dataset(dataset = tasks[t])
    
    for p in CSIC_train.model.classifier.parameters():
        p.requires_grad = False

    # convert back the middle network to final combined network
    if t>0:
        parameter_combined_network = BertClassifier_base()
        parameter_combined_network_dict = parameter_combined_network.bert.state_dict()
        model_dict = CSIC_train.model.bert.module.state_dict()
        state_dict = {}
        # for n,p in model_dict.items():
        #     print(n)

        # construct combination parameters and use them initial standard BERT model.
        for n,p in parameter_combined_network_dict.items():
            if 'word_embeddings' in n or 'position_embeddings' in n or 'token_type_embeddings' in n:
                continue
            elif "LayerNorm" in n:
                if 'weight' in n:
                    state_dict[n] = model_dict[n[:-7]+'_combine.original'+n[-7:]] + model_dict[n[:-7]+'_combine.beta_o']* model_dict[n[:-7]+'_combine.original'+n[-7:]] + model_dict[n[:-7]+'_combine.beta_f']*model_dict[n[:-7]+'_combine.finetuned'+n[-7:]]

                elif 'bias' in n:
                    state_dict[n] = model_dict[n[:-5]+'_combine.original'+n[-5:]] + model_dict[n[:-5]+'_combine.beta_o']* model_dict[n[:-5]+'_combine.original'+n[-5:]] + model_dict[n[:-5]+'_combine.beta_f']*model_dict[n[:-5]+'_combine.finetuned'+n[-5:]]
            else :
                if 'weight' in n:
                    state_dict[n] = model_dict[n[:-7]+'_combine.original'+n[-7:]] + model_dict[n[:-7]+'_combine.alpha_o'].unsqueeze(1)* model_dict[n[:-7]+'_combine.original'+n[-7:]] + model_dict[n[:-7]+'_combine.alpha_f'].unsqueeze(1)*model_dict[n[:-7]+'_combine.finetuned'+n[-7:]]

                elif 'bias' in n:
                    state_dict[n] = model_dict[n[:-5]+'_combine.original'+n[-5:]] + model_dict[n[:-5]+'_combine.alpha_o']* model_dict[n[:-5]+'_combine.original'+n[-5:]] + model_dict[n[:-5]+'_combine.alpha_f']*model_dict[n[:-5]+'_combine.finetuned'+n[-5:]]


        parameter_combined_network_dict.update(state_dict)
        parameter_combined_network.bert.load_state_dict(parameter_combined_network_dict)
        parameter_combined_network.datasets = deepcopy(CSIC_train.model.datasets)
        parameter_combined_network.classifiers = deepcopy(CSIC_train.model.classifiers)
        parameter_combined_network.set_dataset(tasks[t])

        CSIC_train.model = parameter_combined_network
        CSIC_train.model.bert = torch.nn.DataParallel(CSIC_train.model.bert)
        CSIC_train.model = CSIC_train.model.cuda()

    valid_acc_t = {}
    best_avg = 0
    for task in range(t+1):
        CSIC_train.model.set_dataset(CSIC_train.tasks[task])
        valid_loss_t, valid_acc_t[task] = CSIC_train.eval(task, test_loader[task],False)
        best_avg += valid_acc_t[task]
        print('{} test: loss={:.3f}, acc={:5.1f}% |'.format(task,valid_loss_t, 100 * valid_acc_t[task]), end='')
        CSIC_train.logger.add(epoch=(t * CSIC_train.nepochs) +1, task_num=task + 1, test_loss=valid_loss_t,
                        test_acc=valid_acc_t[task])
    
    print('best_avg_Valid:  acc={:5.1f}% |'.format(100 * best_avg/(t+1)), end='')
    CSIC_train.logger.add(task= t, avg_acc =100 * best_avg/(t+1) )
    CSIC_train.logger.save()