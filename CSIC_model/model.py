from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


from torch.nn import CrossEntropyLoss, MSELoss
import sys
from CSIC_model.network import BertModel
from CSIC_model.config_bert import BertConfig
from transformers import BertModel as pretrainmodel
from copy import deepcopy

class BERTCombine(nn.Module):
    def __init__(self, original_network, finetuned_network, alpha_init, make_model=True):
        super(BERTCombine,self).__init__()
        self.original_network = original_network
        self.finetuned_network = finetuned_network
        self.alpha_init = alpha_init
        if make_model:
            self.make_model()

    def make_model(self):

        self.bert = BertModel(BertConfig(output_hidden_states=True,combine = True,combine_LayerNorm = True,alpha_init=self.alpha_init))
        self.datasets, self.classifiers = [], nn.ModuleList()

        # freeze Embedding ,original network and finetuned network
        for n,param in enumerate(self.bert.named_parameters()):
            if n<3:
                param[1].requires_grad = False
            if 'original' in param[0]:
                param[1].requires_grad = False
            if 'finetuned' in param[0]:
                param[1].requires_grad = False
       
        model_dict = self.bert.state_dict()
        old_model_dict = self.original_network.bert.module.state_dict()
        new_model_dict = self.finetuned_network.bert.module.state_dict()
        state_dict = {}
        for (n1,p_old),(n2,p_new) in zip(old_model_dict.items(),new_model_dict.items()):
            if 'word_embeddings' in n1 or 'position_embeddings' in n1 or 'token_type_embeddings' in n1:
                state_dict[n1] = p_old
                continue

            if "weight" in n1:
                state_dict[n1[:-7]+'_combine.original.weight'] = p_old
                state_dict[n2[:-7]+'_combine.finetuned.weight'] = p_new
            elif "bias" in n1:
                state_dict[n1[:-5]+'_combine.original.bias'] = p_old
                state_dict[n2[:-5]+'_combine.finetuned.bias'] = p_new
            else:
                print(n1)

        model_dict.update(state_dict)
        self.bert.load_state_dict(model_dict)



        self.dropout = nn.Dropout(0.1)
        self.classifiers = deepcopy(self.finetuned_network.classifiers)
        self.datasets = deepcopy(self.finetuned_network.datasets)

        self.classifier = None

        
    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(768, num_outputs))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # if self.num_labels == 1:
            #     #  We are doing regression
            #     loss_fct = MSELoss()
            #     loss = loss_fct(logits.view(-1), labels.view(-1))
            # else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClassifier_base(nn.Module):
    def __init__(self, make_model=True):
        super(BertClassifier_base, self).__init__()

        if make_model:
            self.make_model()


    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        self.bert = pretrainmodel.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 2, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
            )
        # self.bert = BertModel(BertConfig(output_hidden_states=True,bayes = False,mult = False,combine = False,combine_LayerNorm = False))

        self.datasets, self.classifiers = [], nn.ModuleList()

        # freeze Embedding and LayerNorm
        for n,param in enumerate(self.bert.named_parameters()):
            if n<3:
                param[1].requires_grad = False
            # if 'LayerNorm' in param[0]:   #'bias' in param[0] or 
            #     param[1].requires_grad = False
        self.dropout = nn.Dropout(0.1)

        self.classifier = None

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(768, num_outputs))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # if self.num_labels == 1:
            #     #  We are doing regression
            #     loss_fct = MSELoss()
            #     loss = loss_fct(logits.view(-1), labels.view(-1))
            # else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
