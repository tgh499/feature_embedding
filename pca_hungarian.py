import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def generate_cost_matrix_for_matlab_munkres(mapping_filename, output_filename):
    X = pd.read_csv(mapping_filename, header=None)
    X = X.values

    pca = PCA(n_components = 2)
    X_pca = pca.fit_transform(X)

    x_min = np.min(X_pca[:,0])
    x_max = np.max(X_pca[:,0])
    y_min = np.min(X_pca[:,1])
    y_max = np.max(X_pca[:,1])

    mapping_coords = []

    for i,j in X_pca:
        temp = []
        temp.append(i-x_min)
        temp.append(j-y_min)
        mapping_coords.append(temp)
        
    mapping_coords = np.array(mapping_coords)    
        
    min_x_coord = np.min(mapping_coords[:,0])
    min_y_coord = np.min(mapping_coords[:,1])    

    max_x = np.max(mapping_coords)
    mapping_coords = mapping_coords/ max_x * 28


    feature_array_2d = np.zeros(shape=(28,28))

    list_of_indices = []

    for row in range(len(feature_array_2d)):
        for col in range(len(feature_array_2d[:,0])):
            list_of_indices.append([row, col])

    cost_matrix = []

    for items in list_of_indices:
        temp = []
        for coords in mapping_coords:
            temp.append(np.linalg.norm(items-coords))
        cost_matrix.append(temp)


    print_data = pd.DataFrame(cost_matrix)
    print_data.to_csv(output_filename, encoding='utf-8', index=False, header=None)

perplexities = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350]
for perplexity in perplexities:
    mapping_filename = "mapping_ecl_" + str(perplexity) + ".csv"
    output_filename = "cost_ecl_" + str(perplexity) + ".csv"
    generate_cost_matrix_for_matlab_munkres(mapping_filename, output_filename)