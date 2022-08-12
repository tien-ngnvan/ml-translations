import tensorflow as tf
import re
import pickle
import numpy as np
from pyvi import ViTokenizer


def process_data(data):
    text = data.strip().lower()
    text = re.sub(r'([!.?])', r' \1', text)
    text = re.sub(r"\s+", r' ', text)
    text = re.sub("&apos;", "'", text)
    text = re.sub("&quot;", "", text)
    text = re.sub("&#", "", text) 

    return text

def vn_tokenizer(data):
    # To perform word segmentation only
    #word_text = annotator.tokenize(data) 
    #text_tmp = []
    #for lines in word_text:
    #    for word in lines:
    #        text_tmp.append(word)
    text_tmp = ViTokenizer.tokenize(data)
       
    #text_tmp = 'start '+ text_tmp + ' end'

    return text_tmp 

def format_dataset(inp_text, inp_tgt):
    
    return ({
        'inp_encoder': inp_text,
        'inp_decoder': inp_tgt[:, :-1]
    }, inp_tgt[:, 1:])

def get_token(text, tokenizer, maxlen, special=False):
    tokens = tokenizer.encode_plus(text, max_length=maxlen, truncation=True,
                                   add_special_tokens=special, padding='max_length', 
                                   return_tensors='tf')
    return tokens['input_ids'][0]

def make_dataset(pairs, source_token, target_token, batch_size, maxlen):
    enc_texts, dec_texts = pairs

    enc_texts = [get_token(text, source_token, maxlen, False) for text in enc_texts]
    dec_texts = [get_token(text, target_token, maxlen, False) for text in dec_texts]
    
    ds = tf.data.Dataset.from_tensor_slices((enc_texts, dec_texts))
    
    ds = ds.shuffle(2048).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds

def make_dataset_eval(pairs, source_token, target_token, batch_size, maxlen):
    enc_texts, dec_texts = pairs

    enc_texts = [get_token(text, source_token, maxlen, False) for text in enc_texts]
    dec_texts = [get_token(text, target_token, maxlen, False) for text in dec_texts]
    
    ds = tf.data.Dataset.from_tensor_slices((enc_texts, dec_texts))
    
    ds = ds.cache().shuffle(2048).prefetch(tf.data.AUTOTUNE)
    
    return ds
    