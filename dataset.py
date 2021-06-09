import collections
import glob
import os
import random
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# from networks import bert_net

def read_files(file_class):
    path = "data/mtl-dataset/"
    train_label,test_label=[],[]
    train_text,test_text = [],[]
    with open(path+file_class+'.train',encoding = 'utf8',errors="ignore") as file_input:
        for line in file_input.readlines():
            train_label.append(int(line.split('\t')[0]))
            train_text.append(line.split('\t')[1])
    with open(path+file_class+'.test',encoding='utf8',errors="ignore") as file_input:
        for line in file_input.readlines():
            test_label.append(int(line.split('\t')[0]))
            test_text.append(line.split('\t')[1])
    return train_label,test_label,train_text,test_text



def bert_train_loader(data_name):
  
    y_train,y_test,train_text,test_text=read_files(data_name)


    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sentences = train_text
    labels = y_train
    test_sentences=test_text
    test_labels=y_test


    MAX_LEN=256

    input_ids = [tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN,truncation=True) for sent in sentences]
    test_input_ids=[tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN,truncation=True) for sent in test_sentences]


    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                            value=0, truncating="post", padding="post")

    test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", 
                            value=0, truncating="post", padding="post")


    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    test_attention_masks = []

    # For each sentence...
    for sent in test_input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        test_attention_masks.append(att_mask)



    # # Use 90% for training and 10% for validation.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                                random_state=2020, test_size=0.125)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                random_state=2020, test_size=0.125)

    train_inputs = torch.LongTensor(train_inputs)
    validation_inputs = torch.LongTensor(validation_inputs)
    test_inputs=torch.LongTensor(test_input_ids)

    train_labels = torch.LongTensor(train_labels)
    validation_labels = torch.LongTensor(validation_labels)
    test_labels=torch.LongTensor(test_labels)

    train_masks = torch.LongTensor(train_masks)
    validation_masks = torch.LongTensor(validation_masks)
    test_masks=torch.LongTensor(test_attention_masks)


    # The DataLoader needs to know our batch size for training, so we specify it 
    # here.
    # For fine-tuning BERT on a specific task, the authors recommend a batch size of
    # 16 or 32.

    batch_size = 16

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Create the DataLoader for our test set.
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader,validation_dataloader, test_dataloader

