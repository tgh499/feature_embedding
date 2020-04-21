#!/ddn/home4/r2444/anaconda3/bin/python
import pandas as pd
import numpy as np
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
import copy
import math

input_filename = 'test_randomized.csv'

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(16):
		distance += (int(row1[i]) - int(row2[i]))**2
	return math.sqrt(distance)

def generate_sixteen_digit_codebook():
    length = int(math.pow(2,16))
    code_words = []
    for i in range(length):
        b = bin(i)[2:].zfill(16)
        code_words.append(b)
    code_words_np = np.array(code_words)
    return(code_words_np)

def generate_patches(image_sample):
    #image_sample_normalized = np.true_divide(image_sample, 255)
    for index, feature in enumerate(image_sample):
        if feature > 0:
            image_sample[index] = 1
        else:
            image_sample[index] = 0

    greyIm = np.reshape(image_sample, (28,28))

    S=greyIm.shape
    N = 2
    patches = []

    for row in range(S[0]):
        for col in range(S[1]):
            Lx=np.max([0,col-N])
            Ux=np.min([S[1],col+N])
            Ly=np.max([0,row-N])
            Uy=np.min([S[0],row+N])
            patch = greyIm[Ly:Uy,Lx:Ux]
            if len(patch) == 4 and len(patch[0]) == 4:
                patches.append(patch)
    return(patches)

def find_nearest_patch(codebook_patches, sample_patches):
    encoded_sample = []
    for i in range(len(sample_patches)):
        patch_distances = []
        for j in range(len(codebook_patches)):
            sample_patch = sample_patches[i].flatten()
            euclid_dist = euclidean_distance(codebook_patches[j], sample_patch)
            patch_distances.append(euclid_dist)
        #print(patch_distances)
        index_min = np.argmin(patch_distances)
        encoded_sample.append(index_min)
    return(encoded_sample)

def find_nearest_patch_universal_dict(sample_patches):
    encoded_sample = []
    for i in range(len(sample_patches)):
        patch_index = 0
        sample_patch = sample_patches[i].flatten()
        sample_patch = sample_patch.astype(int)
        sample_patch_str = ''.join(map(str, sample_patch))
        patch_index = int(sample_patch_str,2)
        #print(patch_distances)
        encoded_sample.append(patch_index)

    return(encoded_sample)

def get_normalized_mapping(perplexity):
    mapping_filename = "mapping_ecl_randomized_" + str(perplexity) + ".csv"
    mapping = pd.read_csv(mapping_filename, header=None)
    mapping = mapping.values

    x_coords = []
    y_coords = []
    for i,j in enumerate(mapping):
        x_coords.append(j[0])
        y_coords.append(j[1])

    if np.max(x_coords) > np.max(y_coords):
        temp = np.max(x_coords) - np.max(y_coords)
        for i,j in enumerate(mapping):
            mapping[i][1] = j[1] + temp
    else:
        temp = np.max(y_coords) - np.max(x_coords)
        for i,j in enumerate(mapping):
            mapping[i][0] = j[0] + temp

    if (np.min(y_coords)< np.min(x_coords)):
        temp = np.min(y_coords)
        if np.min(y_coords) < 0:
            temp = -(temp)
            mapping = mapping + temp
        else:
            mapping = mapping + temp
    else:
        temp = np.min(x_coords)
        if np.min(x_coords) < 0:
            temp = -(temp)
            mapping = mapping + temp
        else:
            mapping = mapping + temp
    return(mapping)


def reorganize_features_according_to_mapping(mapping, data_features):
    x_coords = []
    y_coords = []
    for i,j in enumerate(mapping):
        x_coords.append(j[0])
        y_coords.append(j[1])

    oneD_mapping = []
    dim = 28

    x_coords_max = np.max(x_coords)
    y_coords_max = np.max(y_coords)
    x_coords_min = np.min(x_coords)
    y_coords_min = np.min(y_coords)

    for i in mapping:
        temp = []
        temp0 = (i[0] - x_coords_min) / ((x_coords_max - x_coords_min) / (dim-1))
        temp1 = (i[1] - y_coords_min) / ((y_coords_max - y_coords_min) / (dim-1))
        temp = np.round(temp1) * dim + np.round(temp0)
        oneD_mapping.append(int(temp))

    tsne_mapped_data_features = np.zeros((len(data_features), len(data_features[0])))

    for i,j in enumerate(tsne_mapped_data_features):
        for k,l in enumerate(oneD_mapping):
            tsne_mapped_data_features[i][l] = data_features[i][k]

    return(tsne_mapped_data_features)


data = pd.read_csv(input_filename, header=None)
data = data.head(1000)
features= data.columns[1:]
label = data.columns[0]
data_features = data[features]
data_label = data[label]
data_features_np = data_features.to_numpy()
data_label_np = data_label.to_numpy()
#codebook_patches_np = generate_sixteen_digit_codebook()

perplexities = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350]
for perplexity in perplexities:
    mapping = get_normalized_mapping(perplexity)
    data_features_tsne_mapped = reorganize_features_according_to_mapping(mapping, data_features_np)
    output_filename = "test_quantized_ecl_" + str(perplexity) + ".csv"

    new_dataset = []
    for i in range(len(data_features_tsne_mapped)):
        temp_new_sample = []
        sample_patches = generate_patches(data_features_tsne_mapped[i]) # also gets converted to binary
        temp_new_sample.append(int(data_label_np[i]))
        temp_new_sample += find_nearest_patch_universal_dict(sample_patches)
        new_dataset.append(temp_new_sample)

    new_dataset_pd = pd.DataFrame(new_dataset)
    new_dataset_pd.to_csv(output_filename, encoding='utf-8', index=False, header=None)

'''
codebook_patches = ["000", "001", "010", "011"]

sample_patches = [[0,1,0], [0,0,0], [0,1,1], [0,0,1]]

find_nearest_patch_universal_dict(codebook_patches, sample_patches)
'''
