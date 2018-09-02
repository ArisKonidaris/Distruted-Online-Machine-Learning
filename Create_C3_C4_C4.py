import numpy as np
import pandas as pd
import tensorflow as tf
import Augmentor
import matplotlib.pyplot as plt
import math
import copy
import datetime
import skimage
from skimage.feature import hog

seed = int(datetime.datetime.utcnow().strftime('%m%d%H%M%S'))
np.random.seed(seed)


###############################################################################################
###############################################################################################


# Helper function for loading the MNIST dataset from a .idx3-ubyte file.
def get_data_and_labels(images_filename, labels_filename):
    print("Opening files ...")
    images_file = open(images_filename, "rb")
    labels_file = open(labels_filename, "rb")

    try:
        print("Reading files ...")
        images_file.read(4)
        num_of_items = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_rows = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_colums = int.from_bytes(images_file.read(4), byteorder="big")
        labels_file.read(8)

        num_of_image_values = num_of_rows * num_of_colums
        data = [[None for x in range(num_of_image_values)]
                for y in range(num_of_items)]
        labels = []
        for item in range(num_of_items):
            print("Current image number: %7d" % item)
            for value in range(num_of_image_values):
                data[item][value] = int.from_bytes(images_file.read(1),
                                                   byteorder="big")
            labels.append(int.from_bytes(labels_file.read(1), byteorder="big"))
        return data, labels
    finally:
        images_file.close()
        labels_file.close()
        print("Files closed.")

# A helper function for ploting the MNIST digits.
def plot_images(images, img_shape=(28, 28)):
    assert len(images) == 36

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(6, 6)

    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
# A function for creating a txt file for the given dataset.
def cr_txt_hdf5_format(dset, filename='dataset.txt'):
    f = open(filename,'w')
    print('')
    print('Creating file {}'.format(filename))
    for i in range(dset.shape[0]):
        data=str(i+1)+" "
        for j in range(dset.shape[1]-1):
            data+=str(dset[i][j])+" "
        data+=str(dset[i][dset.shape[1]-1])
        f.write(data);
        f.write("\n");

    f.close()


###############################################################################################
###############################################################################################


### Reading the original MNIST dataset. ###
train = True
batch_size = 128
dataset_sizes = 0

if(train):
    dataset_sizes = 1000000
    images_filename = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/train-images.idx3-ubyte"
    labels_filename = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/train-labels.idx1-ubyte"
else:
    dataset_sizes = 20000
    images_filename = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/t10k-images.idx3-ubyte"
    labels_filename = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/t10k-labels.idx1-ubyte"

data, labels = get_data_and_labels(images_filename, labels_filename)
train_images = np.array(data)
del data
train_labels = np.array(labels)
train_labels = np.eye(10)[train_labels]
del labels


###############################################################################################
###############################################################################################


### The main parameters of the dataset. ###

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

# Number of training examples
num_train_exms = np.shape(train_images)[0]

# Number of training labels
num_train_lbs = np.shape(train_labels)[0]

# Number of generated augmented images for each image
aug_per_im = 36

# Number of augmented data points
num_aug_pts = num_train_exms*aug_per_im

# Size of final augmented dataset
aug_data_sz = num_aug_pts+num_train_exms


###############################################################################################
###############################################################################################


### Initializing the hash map, the digit buckets and the dataset generators###

TrainSet = np.hstack((train_images, np.argmax(train_labels,1).reshape(train_images.shape[0],1)+1)).astype(np.float32)
dataframe = pd.DataFrame(data=TrainSet)
dataframe = dataframe.sort_values(by=[img_size_flat], kind='quicksort')
del TrainSet, train_images, train_labels 

backets = {}
gen = {}
for i in range(num_classes):
    
    backets[i] = np.array(dataframe[dataframe[img_size_flat]==float(i+1)].values)
    
    gen[i] = Augmentor.Pipeline()
    gen[i].set_seed(seed+1+i)
    gen[i].random_distortion(probability=0.45, grid_width=4, grid_height=4, magnitude=1)
    if(train):
        gen[i].rotate_without_crop(probability=1., max_left_rotation=-15, max_right_rotation=15, expand=False)
        gen[i].shear(probability=0.5, max_shear_left=8, max_shear_right=8)
        gen[i].skew(probability=0.5, magnitude=0.2)
    else:
        gen[i].rotate_without_crop(probability=1., max_left_rotation=-3, max_right_rotation=3, expand=False)
        gen[i].shear(probability=0.5, max_shear_left=4, max_shear_right=4)
        gen[i].skew(probability=0.5, magnitude=0.1)
    gen[i].resize(probability=1., width=img_size, height=img_size)
    gen[i].greyscale(probability=1.)
    gen[i] = gen[i].keras_generator_from_array(np.reshape(backets[i][:,0:img_size_flat], (np.shape(backets[i])[0], 28, 28, 1)),
                                               backets[i][:,img_size_flat], batch_size=batch_size)
        
del dataframe


###############################################################################################
###############################################################################################


## Creating the C3 dataset ##
pos = 0
C3 = np.zeros((dataset_sizes,img_size_flat+1), dtype=np.float32)   
samples = np.random.choice(np.arange(1, 7), dataset_sizes, p=(1/6)*np.ones((6,)))
for i in range(6):
    
    print(pos)
    num_of_dgts = np.sum(samples == (1+i)*np.ones(np.shape(samples)).astype(np.int32))
    pointer = 0
    digits = np.zeros((num_of_dgts,img_size_flat+1), dtype=np.float32)
    while True:
            
        if (pointer + batch_size > num_of_dgts):
            batch = num_of_dgts % batch_size
        else:
            batch = batch_size
                
        X, Y = next(gen[i])
        X = np.reshape(X, (np.shape(X)[0], img_size_flat))
        X = 255.*X[0:batch]
        Y = Y[0:batch]
            
        digits[pointer:pointer+batch,:] =  np.hstack((X, Y.reshape(batch,1)))
            
        pointer += batch
        if(pointer==num_of_dgts):
            break
       
    digits[0:np.shape(backets[i])[0],:] = backets[i]
    C3[pos:pos+num_of_dgts,:] = digits
    pos += num_of_dgts
C3[:,0:img_size_flat] /= 255

## Adding the HOG discriptors to the C3 dataset ##
C3_HOGS = np.zeros((dataset_sizes,144), dtype=np.float32)
for i in range(np.shape(C3)[0]):
    hog_feature, _ = hog((255*C3[i,0:img_size_flat].reshape(img_shape)).astype(np.uint8), \
                         orientations=9, pixels_per_cell=(7, 7), cells_per_block=(4,4), \
                         feature_vector=True, visualise=True, block_norm='L2')
    C3_HOGS[i,:] = np.array(hog_feature)

## Concatenating the original dataset with the HOG descriptors ##
C3_ = np.hstack((C3[:,0:img_size_flat],C3_HOGS))
C3_ = np.hstack((C3_,C3[:,img_size_flat].reshape(np.shape(C3)[0],1)))


###############################################################################################
###############################################################################################


## Creating the C4 dataset ##
pos = 0
C4 = np.zeros((dataset_sizes,img_size_flat+1), dtype=np.float32)   
samples = np.random.choice(5+np.arange(1, 5), dataset_sizes, p=(1/4)*np.ones((4,)))
for i in range(6,10):
    
    print(pos)
    num_of_dgts = np.sum(samples == i*np.ones(np.shape(samples)).astype(np.int32))
    pointer = 0
    digits = np.zeros((num_of_dgts,img_size_flat+1), dtype=np.float32)
    while True:
            
        if (pointer + batch_size > num_of_dgts):
            batch = num_of_dgts % batch_size
        else:
            batch = batch_size
                
        X, Y = next(gen[i])
        X = np.reshape(X, (np.shape(X)[0], img_size_flat))
        X = 255.*X[0:batch]
        Y = Y[0:batch]
            
        digits[pointer:pointer+batch,:] =  np.hstack((X, Y.reshape(batch,1)))
            
        pointer += batch
        if(pointer==num_of_dgts):
            break
       
    digits[0:np.shape(backets[i])[0],:] = backets[i]
    C4[pos:pos+num_of_dgts,:] = digits
    pos += num_of_dgts
C4[:,0:img_size_flat] /= 255

## Adding the HOG discriptors to the C5 dataset ##
C4_HOGS = np.zeros((dataset_sizes,144), dtype=np.float32)
for i in range(np.shape(C4)[0]):
    hog_feature, _ = hog((255*C4[i,0:img_size_flat].reshape(img_shape)).astype(np.uint8), \
                         orientations=9, pixels_per_cell=(7, 7), cells_per_block=(4,4), \
                         feature_vector=True, visualise=True, block_norm='L2')
    C4_HOGS[i,:] = np.array(hog_feature)

## Concatenating the original dataset with the HOG descriptors ##
C4_ = np.hstack((C4[:,0:img_size_flat],C4_HOGS))
C4_ = np.hstack((C4_,C4[:,img_size_flat].reshape(np.shape(C4)[0],1)))


###############################################################################################
###############################################################################################


## Creating the C5 dataset ##
pos = 0
C5 = np.zeros((dataset_sizes,img_size_flat+1), dtype=np.float32)  
samples = np.random.choice(np.arange(1, 11), dataset_sizes, p=(1/10)*np.ones((10,)))
for i in range(10):
    
    print(pos)
    num_of_dgts = np.sum(samples == (1+i)*np.ones(np.shape(samples)).astype(np.int32))
    pointer = 0
    digits = np.zeros((num_of_dgts,img_size_flat+1), dtype=np.float32)
    while True:
            
        if (pointer + batch_size > num_of_dgts):
            batch = num_of_dgts % batch_size
        else:
            batch = batch_size
                
        X, Y = next(gen[i])
        X = np.reshape(X, (np.shape(X)[0], img_size_flat))
        X = 255.*X[0:batch]
        Y = Y[0:batch]
            
        digits[pointer:pointer+batch,:] =  np.hstack((X, Y.reshape(batch,1)))
            
        pointer += batch
        if(pointer==num_of_dgts):
            break
    
    C5[pos:pos+num_of_dgts,:] = digits
    pos += num_of_dgts
C5[:,0:img_size_flat] /= 255

## Adding the HOG discriptors to the C5 dataset ##
C5_HOGS = np.zeros((dataset_sizes,144), dtype=np.float32)
for i in range(np.shape(C5)[0]):
    hog_feature, _ = hog((255*C5[i,0:img_size_flat].reshape(img_shape)).astype(np.uint8), \
                         orientations=9, pixels_per_cell=(7, 7), cells_per_block=(4,4), \
                         feature_vector=True, visualise=True, block_norm='L2')
    C5_HOGS[i,:] = np.array(hog_feature)

## Concatenating the original dataset with the HOG descriptors ##
C5_ = np.hstack((C5[:,0:img_size_flat],C5_HOGS))
C5_ = np.hstack((C5_,C5[:,img_size_flat].reshape(np.shape(C5)[0],1)))


###############################################################################################
###############################################################################################


## Write C3 dataset into a text file with HDF5 format ##
random_suffle = np.random.choice(np.shape(C3_)[0], np.shape(C3_)[0], replace=False)
C3_ = C3_[random_suffle]
random_suffle = np.random.choice(np.shape(C3_)[0], np.shape(C3_)[0], replace=False)
C3_ = C3_[random_suffle]
random_suffle = np.random.choice(np.shape(C3_)[0], np.shape(C3_)[0], replace=False)
C3_ = C3_[random_suffle]

if(train):
    filename = 'C3_Train.txt'
else:
    filename = 'C3_Test.txt'

cr_txt_hdf5_format(C3_, filename)


## Write C4 dataset into a text file with HDF5 format ##
random_suffle = np.random.choice(np.shape(C4_)[0], np.shape(C4_)[0], replace=False)
C4_ = C4_[random_suffle]
random_suffle = np.random.choice(np.shape(C4_)[0], np.shape(C4_)[0], replace=False)
C4_ = C4_[random_suffle]
random_suffle = np.random.choice(np.shape(C4_)[0], np.shape(C4_)[0], replace=False)
C4_ = C4_[random_suffle]

if(train):
    filename = 'C4_Train.txt'
else:
    filename = 'C4_Test.txt'

cr_txt_hdf5_format(C4_, filename)


## Write C5 dataset into a text file with HDF5 format ##
random_suffle = np.random.choice(np.shape(C5_)[0], np.shape(C5_)[0], replace=False)
C5_ = C5_[random_suffle]
random_suffle = np.random.choice(np.shape(C5_)[0], np.shape(C5_)[0], replace=False)
C5_ = C5_[random_suffle]
random_suffle = np.random.choice(np.shape(C5_)[0], np.shape(C5_)[0], replace=False)
C5_ = C5_[random_suffle]

if(train):
    filename = 'C5_Train.txt'
else:
    filename = 'C5_Test.txt'

cr_txt_hdf5_format(C5_, filename)