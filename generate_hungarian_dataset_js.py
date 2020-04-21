#!/ddn/home4/r2444/anaconda3/bin/python
import pandas as pd
import numpy as np
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
import copy
import math


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
        sample_patch_str = ''.join(map(str, sample_patch))
        patch_index = int(sample_patch_str,2)
        #print(patch_distances)
        encoded_sample.append(patch_index)

    return(encoded_sample)


def reorganize_dataset(input_filename_prefix):
    input_filename = input_filename_prefix + ".csv"
    data = pd.read_csv(input_filename, header=None)
    #data = data.head(1000)
    label = data.columns[0]
    data_label = data[label]
    data_label_np = data_label.to_numpy()
    #codebook_patches_np = generate_sixteen_digit_codebook()

    perplexities = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350]
    #perplexities = [270]
    for perplexity in perplexities:
        index_filename = "rearranged_js_" + str(perplexity) + ".csv"
        rearranged_features = np.loadtxt(index_filename, delimiter=',', unpack=True)
        data_features = data[rearranged_features]
        #data_features_np = data_features.values
        output_filename = input_filename_prefix + "_js_hungarian_" + str(perplexity) + ".csv"

        result = pd.concat([data_label, data_features], axis=1)
        result.to_csv(output_filename, encoding='utf-8', index=False, header=None)

reorganize_dataset('test_randomized')
reorganize_dataset('train_randomized')