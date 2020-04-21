#!/ddn/home4/r2444/anaconda3/bin/python
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def generate_tsne_mapping(X, perplexity, suffix):

    fileName = "mapping_js_" + suffix + str(perplexity) + ".csv"
    X = X.values

    # metric = precomputed, x= distance_matrix <- JS <- KL (x or X_train?)
    #X_embedded = TSNE(n_components=2, perplexity=perplexity, verbose=1, random_state=1).fit_transform(X)
    X_embedded = TSNE(n_components=2, perplexity=perplexity, metric='precomputed', verbose=1, random_state=1).fit_transform(X)

    X_embedded = pd.DataFrame(X_embedded)
    X_embedded.to_csv(fileName, encoding='utf-8', index=False, header=None)


perplexities = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350]
suffix = ""
data = pd.read_csv('distance_matrix_randomized.csv', header=None)
data_np = data.values


for perplexity in perplexities:
    generate_tsne_mapping(data, perplexity, suffix)
