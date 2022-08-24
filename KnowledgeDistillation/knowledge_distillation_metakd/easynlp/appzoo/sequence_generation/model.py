import torch
import numpy as np
import json
import os
import re
from ...modelzoo import AutoConfig, AutoModel,AutoTokenizer,AutoModelForSeq2SeqLM,BertTokenizer,GPT2LMHeadModel
from ..application import Application

def sequence_padding(inputs, length=None, padding=0):
    """Padding the sequence to same length
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')

class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: x.split(" "), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens

class SequenceGeneration(Application):
    """ BERT Classification/Regression Teacher """

    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path,user_defined_parameters=None, **kwargs):
        instance=SequenceGeneration(pretrained_model_name_or_path,user_defined_parameters, **kwargs)
        instance._model._tokenizer=instance._tokenizer
        instance._model._is_zh=instance._is_zh
        return instance._model

    def __init__(self,pretrained_model_name_or_path,user_defined_parameters=None, **kwargs):
        super().__init__()

        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        self.model_name = "sequence_generation"
        if user_defined_parameters is not None:
            if type(user_defined_parameters)=='str':
                self.user_defined_parameters=json.loads(user_defined_parameters)
            else:
                self.user_defined_parameters=user_defined_parameters
        else:
            self.user_defined_parameters={}
        if os.path.exists(pretrained_model_name_or_path):
            local_path=pretrained_model_name_or_path
        else:
            local_path=os.environ['HOME']+'/.easynlp/modelzoo/huggingface/'+pretrained_model_name_or_path

        config_path=local_path+'/config.json'
        self._is_zh=False
        self.is_gpt2=self.user_defined_parameters.get('service')=='chat'
        if self.is_gpt2:
            self._tokenizer = BertTokenizer(vocab_file=local_path+'/vocab.txt', sep_token="[SEP]", 
                                            pad_token="[PAD]", cls_token="[CLS]")

            # Map to cpu device when gpu is not available
            try:
                model_state_dict = torch.load(local_path+'/pytorch_model.bin')
            except RuntimeError:
                model_state_dict = torch.load(local_path+'/pytorch_model.bin',
                                              map_location=torch.device('cpu'))
            
            state_dict_without_prefix = {}
            for key, value in model_state_dict.items():
                key=key.replace('_model.transformer.','').replace('_model.','')
                state_dict_without_prefix[key] = value
            self._model=GPT2LMHeadModel.from_pretrained(local_path,state_dict=state_dict_without_prefix)
            self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            self._is_zh=True
        else:
            with open(config_path,'r') as load_f:
                load_dict = json.load(load_f)
                if ("architectures" in load_dict) and (load_dict["architectures"][0]=='T5ForConditionalGeneration'):
                    self._is_zh=True
                    tokenizer_class=T5PegasusTokenizer
                else:
                    tokenizer_class=AutoTokenizer
                self.tokenizer_class=tokenizer_class  
            self._tokenizer = tokenizer_class.from_pretrained(local_path)
            if self._is_zh:
                self.vocab_idx={}
                with open(local_path+'/vocab.txt','r') as voc:
                    for idx,line in enumerate(voc.readlines()):
                        if idx<=105:
                            continue
                        self.vocab_idx[idx]=line.replace("\n",'')
            else:
                self.vocab_idx={}
                sp_tokenizer =tokenizer_class.from_pretrained(pretrained_model_name_or_path)
                sp_vocab=sp_tokenizer.get_vocab()
                for vocab_key,vocab_val in enumerate(sp_vocab):
                    self.vocab_idx[vocab_key]=vocab_val

            # Map to cpu device when gpu is not available
            try:
                model_state_dict = torch.load(local_path+'/pytorch_model.bin')
            except RuntimeError:
                model_state_dict = torch.load(local_path+'/pytorch_model.bin',
                                              map_location=torch.device('cpu'))
            
            state_dict_without_prefix = {}
            for key, value in model_state_dict.items():
                key=key.replace('xsum.','').replace('mt5.','').replace('_model.','')
                state_dict_without_prefix[key] = value
            self._model=AutoModelForSeq2SeqLM.from_pretrained(local_path,state_dict=state_dict_without_prefix)
            self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, inputs):
        if self.is_gpt2:
            prob = self._model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"])[0]            
        else:
            prob = self._model(input_ids=inputs["input_ids"],
                        decoder_input_ids=inputs["decoder_input_ids"],
                        attention_mask=inputs["attention_mask"],
                        decoder_attention_mask=inputs["decoder_attention_mask"])[0]
        slice_len=prob.size()[1]
        label_len=inputs['decoder_attention_mask'].size()[1]
        if label_len<slice_len:
            slice_len=label_len
        prob = prob[:, :slice_len-1]
        mask = inputs['decoder_attention_mask'][:, 1:slice_len].reshape(-1).bool()
        prob = prob.reshape((-1, prob.size(-1)))[mask]
        labels = inputs['decoder_input_ids'][:, 1:slice_len].reshape(-1)[mask]
        return {
            "prob": prob,
            "labels": labels
        }

    def generate(self,
                 input_ids,
                 attention_mask,
                 num_beams=1,
                 min_length=64,
                 max_length=256,
                 early_stopping=True,
                 no_repeat_ngram_size=2,
                 num_return_sequences=5,
                 decoder_start_token_id=101,
                 eos_token_id=102):
        
        copy_flag=False
        if ("copy" in self.user_defined_parameters) and (self.user_defined_parameters['copy']==True):
            copy_flag=True

        gen_con=[]
        for index,input_one in enumerate(input_ids):
            tmp_filter={}
            for idx in input_one:
                tmp_filter[idx.item()]=True
            
            if copy_flag==True:
                new_bad=[]
                for k in self.vocab_idx:
                    if k in tmp_filter:
                        continue
                    new_bad.append(k)
                new_bad=[new_bad]
            else:
                new_bad=None
            gen = self._model.generate(input_ids=input_ids[index:index+1],
                                    attention_mask=attention_mask[index:index+1],
                                    min_length=min_length,
                                    max_length=max_length,
                                    num_beams=num_beams,
                                    early_stopping=early_stopping,
                                    no_repeat_ngram_size=no_repeat_ngram_size,
                                    num_return_sequences=num_return_sequences,
                                    eos_token_id=eos_token_id,
                                    decoder_start_token_id=decoder_start_token_id,
                                    bad_words_ids=new_bad)
            gen_con.extend(gen.cpu().tolist())

        return gen_con

    def compute_loss(self, model_outputs, inputs):
        loss = self.loss_fct(model_outputs["prob"], model_outputs["labels"])
        return {
            "loss": loss
        }
