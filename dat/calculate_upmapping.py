import json
import re
import pickle
from pprint import pprint
import random
import glob
from scipy import stats
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
embeddings = model.get_input_embeddings()
embeddings = embeddings.weight.detach().cpu().numpy()

pca = PCA(n_components=512)
pca.fit(embeddings)
print(pca.explained_variance_ratio_.sum())

# save the pca.components_ to a file in redteaming_exp
with open("redteaming_exp/llama3_8B_embed_pcas.pkl", "wb") as f:
    pickle.dump(pca.components_, f)