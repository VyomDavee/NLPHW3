import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_data = []
        try:
            with open(nl_path, 'r') as f:
                nl_data = [line.strip() for line in f]
        except FileNotFoundError:
            print(f"Warning: {nl_path} not found")
            return []
        sql_data = []
        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            try:
                with open(sql_path, 'r') as f:
                    sql_data = [line.strip() for line in f]
            except FileNotFoundError:
                print(f"Warning: {sql_path} not found")
                return []
        
        processed_data = []
        for i in range(len(nl_data)):
            item = {'text': nl_data[i]}
            
            # Tokenize NL instruction for encoder
            encoder_tokens = tokenizer(nl_data[i], return_tensors="pt", truncation=True).input_ids.squeeze(0)
            item['encoder_ids'] = encoder_tokens
            item['encoder_mask'] = torch.ones_like(encoder_tokens)
            
            if split != "test":
                # Tokenize SQL query for decoder
                decoder_tokens = tokenizer(sql_data[i], return_tensors="pt", truncation=True).input_ids.squeeze(0)
                item['decoder_tokens'] = decoder_tokens
                item['sql'] = sql_data[i]
            
            processed_data.append(item)
        
        return processed_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [item['encoder_ids'] for item in batch]
    encoder_mask = [item['encoder_mask'] for item in batch]
    decoder_tokens = [item['decoder_tokens'] for item in batch]

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=0)

    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    bos_token_id = tokenizer.get_vocab()['<extra_id_0>']

    decoder_inputs = []
    decoder_targets = []
    initial_decoder_inputs = []

    for tokens in decoder_tokens:
        decoder_input = torch.cat([torch.tensor([bos_token_id]), tokens[:-1]])
        decoder_inputs.append(decoder_input)
        
        decoder_targets.append(tokens)
        
        initial_decoder_inputs.append(torch.tensor([bos_token_id]))
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = pad_sequence(initial_decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [item['encoder_ids'] for item in batch]
    encoder_mask = [item['encoder_mask'] for item in batch]
    

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=0)
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    bos_token_id = tokenizer.get_vocab()['<extra_id_0>']

    initial_decoder_inputs = []
    for _ in range(len(batch)):
        initial_decoder_inputs.append(torch.tensor([bos_token_id]))
    
    initial_decoder_inputs = pad_sequence(initial_decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # Load training data
    train_nl_path = os.path.join(data_folder, "train.nl")
    train_sql_path = os.path.join(data_folder, "train.sql")
    train_x = load_lines(train_nl_path)
    train_y = load_lines(train_sql_path)
    
    # Load development data
    dev_nl_path = os.path.join(data_folder, "dev.nl")
    dev_sql_path = os.path.join(data_folder, "dev.sql")
    dev_x = load_lines(dev_nl_path)
    dev_y = load_lines(dev_sql_path)
    
    # Load test data
    test_nl_path = os.path.join(data_folder, "test.nl")
    test_x = load_lines(test_nl_path)
    
    return train_x, train_y, dev_x, dev_y, test_x