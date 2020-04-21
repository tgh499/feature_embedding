import math
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    kl = np.sum(np.where(a != 0, a * np.log(a / b), 0))
    return(kl)

def JS(x,y):
	mean = 1.0/2*(x+y)
	js=math.sqrt(1.0/2*(KL(x,mean)+KL(y,mean)))
	return(js)


data = pd.read_csv('train_randomized.csv', header=None)
features = data.columns[1:]
label = data.columns[0]
data_features = data[features]
data_label = data[label]

data_features_T = data_features.T # transpose the dataset
data_features_T_np = data_features_T.values

data_features_T_np = np.where(data_features_T_np==0, 0.000001, data_features_T_np) 
                            # can't have 0, so replace with a small value

for i,feature_values in enumerate(data_features_T_np):
    data_features_T_np[i] = np.true_divide(feature_values, np.sum(feature_values))
                            # rows must sum to 1

distance_matrix = np.zeros((len(data_features_T_np), len(data_features_T_np)))

for i,j in enumerate(distance_matrix):
    for k,l in enumerate(j):
        distance_matrix[i][k] = JS(data_features_T_np[i], data_features_T_np[k])

distance_matrix = pd.DataFrame(distance_matrix)
print(distance_matrix.head(50))
distance_matrix.to_csv('distance_matrix_randomized.csv', encoding='utf-8', index=False, header=None)