from sklearn import datasets
import numpy as np
import h5py

f=h5py.File("/home/aris/Desktop/Starting_Cpp_Developing/my_test_dataset.hdf5", "r")

print(f.keys())

#data = f["Linear_datasets/dataset1"]
data = f.get("/Linear_datasets/dataset1")

print("")
print("My Linear Dataset !!!")
print("")
print(data)
print("")
print("shape and type")
print(data.shape)
print(data.dtype)
print("")
print("data value")
print(data.value)

f.close