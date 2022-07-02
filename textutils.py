import os, numpy as np
from time import time

from transformers import BertTokenizer, BertModel
from fse import Vectors, Average, IndexedList
import torch
from config import getopt

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.preprocessing import normalize

import re
#from gensim.models import KeyedVectors as Word2Vec
#from sklearn.preprocessing import normalize
#from nltk.corpus import wordnet as wn
#from nltk.stem.wordnet import WordNetLemmatizer

def classes2embedding(dataset_name, class_name_inputs, opt, w2v=False):
    if dataset_name == 'ucf101':
        clean_class_names = clean_ucf
    elif dataset_name == 'hmdb51':
        clean_class_names = clean_hmdb
    elif dataset_name == 'kinetics':
        clean_class_names = clean_kinetics
    elif dataset_name == 'activitynet':
        clean_class_names = clean_activitynet
    elif dataset_name == 'sun':
        clean_class_names = clean_sun


    #print("dataset name", dataset_name, "class name inputs", class_name_inputs)
    clean_class_inputs = clean_class_names(class_name_inputs, opt, w2v=w2v)


    #OVerrise for w2v, so we can filter classes correctly
    if w2v:
        embedding = w2vavg_embed(clean_class_inputs, opt)
    elif opt.text_embed == 'bert':
        embedding = bert_embed(clean_class_inputs, opt)
    elif opt.text_embed == 'bertavg':
        embedding = bertavg_embed(clean_class_inputs, opt)
    elif opt.text_embed == 'w2vavg':
        embedding = w2vavg_embed(clean_class_inputs, opt)
    
    #embedding = [one_class2embed(class_name, wv_model)[0] for class_name in class_name_inputs]
    #embedding = np.stack(embedding)
    # np.savez('/workplace/data/motion_efs/home/biagib/ZeroShot/W2V_embedding/'+dataset_name,
    #          names=dataset_name, embedding=embedding)
    #return normalize(embedding.squeeze())

    return embedding

def clean_ucf(class_name_inputs, opt, w2v=False):
    #print("+++++++++++++++Inside of cleanucf++++++++++++++++++++++++")
    change = {
        'CleanAndJerk': 'WeightLift',
        'Skijet': 'Jetski',
        'HandStandPushups': 'HandstandPushups',
        'HandstandPushups': 'HandstandPushups',
        'PushUps': 'Pushups',
        'PullUps': 'Pullups',
        'WalkingWithDog' : 'WalkDog',
        'ThrowDiscus': 'ThrowDisc',
        'TaiChi': 'Taichi',
        'CuttingInKitchen': 'CutKitchen',
        'YoYo': 'Yoyo'
    }

    cleaned_class_names = []

    for i in range(len(class_name_inputs)):
        if class_name_inputs[i] in change:
            new_name = change[class_name_inputs[i]]
        else:
            new_name = class_name_inputs[i]
        full_name = re.sub('([A-Z])', r' \1', new_name)[1:].lower()
        #print(full_name)

        if w2v or opt.text_embed in ['w2v', 'w2vavg']:
            cleaned_class_names.append(full_name)
        else:
            cleaned_class_names.append(opt.prompt + full_name)

    #print("The cleaned UCF class names look like", cleaned_class_names)
    return cleaned_class_names

def clean_hmdb(class_name_inputs, opt, w2v=False):
    change = {'clap': 'clapping'}
    cleaned_class_names = []

    for i in range(len(class_name_inputs)):
        if class_name_inputs[i] in change:
            new_name = change[class_name_inputs[i]]
        else:
            new_name = class_name_inputs[i]
        #cleaned_class_names.append(verbs2basicform(new_name.split()))
        if w2v or opt.text_embed in ['w2v', 'w2vavg']:
            cleaned_class_names.append(new_name)
        else:
            cleaned_class_names.append(opt.prompt +  new_name)
    return cleaned_class_names


def clean_kinetics(class_name_inputs, opt, w2v=False):
    change = {
        'clean and jerk': 'weight lift',
        'dancing gangnam style': 'dance korean',
        'breading or breadcrumbing': 'bread crumb',
        'eating doughnuts': 'eat bun',
        'faceplanting': 'fall on face',
        'hoverboarding': 'electric skateboard',
        'hurling (sport)': 'hurl sport',
        'jumpstyle dancing': 'jumping dance',
        'passing American football (in game)': 'american football pass in game',
        'passing American football (not in game)': 'american football pass in park',
        'petting animal (not cat)': 'animal pet',
        'punching person (boxing)': 'punching person in boxing',
        's head": 1}': 'head',
        'shooting goal (soccer)': 'shooting soccer goal',
        'skiing (not slalom or crosscountry)': 'ski',
        'throwing axe': 'throwing ax',
        'tying knot (not on a tie)': 'tie knot',
        'using remote controller (not gaming)': 'remote control',
        'backflip (human)': 'human backflip',
        'blowdrying hair': 'drying hair',
        'making paper aeroplanes': 'make paper airplane',
        'mixing colours': 'mixing colors',
        'photobombing': 'taking pictures',
        'playing rubiks cube': 'playing with rubiks cube',
        'throwing ball (not baseball or American football)': 'throwing ball',
        'curling (sport)': 'curling sport',
        "massaging person's head": 'massaging head',
        'hug (not baby)' : 'hug not baby',
        'bouncing ball (not juggling)': 'bouncing ball not juggling',
        'carve wood with a knife' : 'carve wood with knife'#['massaging', 'head']  # Added
    }
    cleaned_class_names = []

    for i in range(len(class_name_inputs)):
        if class_name_inputs[i] in change:
            new_name = change[class_name_inputs[i]]
        else:
            new_name = class_name_inputs[i]
        #cleaned_class_names.append(verbs2basicform(new_name.split()))
        if w2v or opt.text_embed in ['w2v', 'w2vavg']:
            cleaned_class_names.append(new_name)
        else:
            cleaned_class_names.append(opt.prompt +  new_name)
    
    #print('Here come the completed kinectics classes')
    return cleaned_class_names
def clean_activitynet(class_name_inputs, opt):
    return 0
def clean_sun(class_name_inputs, opt):
    return 0

def bert_embed(class_name_inputs, opt):


    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    tokens = tokenizer(class_name_inputs, return_tensors='pt', padding='max_length', max_length=20)
    outputs = model(**tokens).last_hidden_state.detach().numpy()

    print("The output shape is", outputs.shape)
    return np.stack(outputs)

def bertavg_embed(class_name_inputs, opt):


    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    tokens = tokenizer(class_name_inputs, return_tensors='pt', padding='max_length', max_length=20)
    outputs = model(**tokens).pooler_output.detach().numpy()

    return np.expand_dims(np.stack(outputs), axis=1)

# Method for the word2vec average method
def w2vavg_embed(class_name_inputs, opt):

    class_name_inputs = [x.split(" ") for x in class_name_inputs]

    vecs = Vectors.from_pretrained("word2vec-google-news-300")

    remove = ['a', 'the', 'of', ' ', '', 'and', 'at', 'on', 'in', 'an', 'or',
            'do', 'using', 'with', 'to']

    outputs = []
    for name in class_name_inputs:

        # Clean name spesifically for w2v
        name = [n for n in name if n not in remove]
        not_id = [i for i, n in enumerate(name) if n == '(not']
        if len(not_id) > 0:
                name = name[:not_id[0]]
        name = [name.replace('(', '').replace(')', '') for name in name]
        name = verbs2basicform(name)

        #print(name)
        full = vecs[name]
        avg = full.mean(0)
        outputs.append(avg)
    print("before normalization we have", np.stack(outputs).shape)
    outputs = normalize(np.stack(outputs))
    #outputs = np.stack(outputs)

    return np.expand_dims(outputs, axis=1)

def verbs2basicform(words):
    ret = []
    for w in words:
        analysis = wn.synsets(w)
        if any([a.pos() == 'v' for a in analysis]):
            w = WordNetLemmatizer().lemmatize(w, 'v')
        ret.append(w)
    return ret

if __name__ == "__main__":
    opt = getopt()
    examples = ["testing the first class", "testing the second class"]

    print(w2v_embed(examples, opt))