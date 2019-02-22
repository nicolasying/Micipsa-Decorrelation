# Songsheng YING
# Micipsa, Embedding Decorrelation

# Python 3.6

import os, logging, gensim
import numpy as np
import pandas as pd
from sklearn import linear_model

# Global configuration
lang = 'French'
root_path = '/home/sying/Documents/Decorrelation/' + lang + '/'

if lang == 'French':
    sim_model_path = root_path + 'wolf_15k_850d.txt'
    mix_model_path = root_path + 'depglove_200d_eric.txt'
    asn_model_path = root_path + 'asn_embedding.txt'
    sig_model_path = root_path + 'sig_embedding.txt'
elif lang == 'English':
    sim_model_path = root_path + 'sim_embedding.txt'
    mix_model_path = root_path + 'glove.840B.300d.txt'
    asn_model_path = root_path + 'asn_embedding.txt'
    sig_model_path = root_path + 'sig_embedding.txt'

# Word Alignment in two embeddings
file = open(sim_model_path, mode='r')
voc, dim = file.readline().split(' ')
voc = int(voc)
dim_sim = int(dim)

file_data = file.readlines()
file.close()

vocabulary = dict()
idx = 0
sim_model = np.zeros((voc, dim_sim), dtype=np.float32)
word_list = []

for line in file_data:
    vector = line.split(' ')
    word_list.append(vector[0])
    vocabulary[vector[0]] = idx
    sim_model[idx,:] = vector[1:]
    idx += 1

file = open(mix_model_path, mode='r', encoding="cp1252")
file_data = file.readlines()
file.close()

dim_mix = len(file_data[0].split(' ')) - 1
mix_model = np.zeros((voc, dim_mix), dtype=np.float32)

for line in file_data:
    vector = line.split(' ')
    try:
        idx = vocabulary.pop(vector[0])
        mix_model[idx,:] = vector[1:]
    except KeyError:
        continue


vocabulary_mask = np.ones(voc, dtype=np.bool)
vocabulary_mask[list(vocabulary.values())] = 0

sim_model_red = sim_model[vocabulary_mask,:]
mix_model_red = mix_model[vocabulary_mask,:]

# Linear Decorrelation by projecting A emb. onto B emb.

reg = linear_model.LinearRegression()
reg.fit(sim_model_red, mix_model_red)
score = reg.score(sim_model_red, mix_model_red)
print("Model mapping score: ", score)

asn_model_red = mix_model_red - reg.predict(sim_model_red)
word_list = [word_list[i] for i in range(len(word_list)) if vocabulary_mask[i]]
file = open(asn_model_path, mode='w')
file.write('{} {}\n'.format(len(word_list), dim_mix))
asn_model_df = pd.DataFrame(asn_model_red, index=word_list)
file.write(asn_model_df.to_csv(sep=' ', header=False))
file.close()

file = open(sig_model_path, mode='w')
file.write('{} {}\n'.format(len(word_list), dim_mix))
sig_model_red = reg.predict(sim_model_red)
sig_model_df = pd.DataFrame(sig_model_red, index=word_list)
file.write(sig_model_df.to_csv(sep=' ', header=False))
file.close()
