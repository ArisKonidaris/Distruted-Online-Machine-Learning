from sklearn import datasets
import numpy as np
import h5py

NMR_OF_SAMPLES = 1000000
NMR_OF_FEAT = 100

# Set random seed (for reproducibility)
np.random.seed(1000)

#dataset=datasets.make_regression(n_samples=NMR_OF_SAMPLES, n_features=NMR_OF_FEAT)

dataset=datasets.make_classification(n_samples=NMR_OF_SAMPLES,
				 n_features=NMR_OF_FEAT,
				 n_informative=NMR_OF_FEAT-15,
				 n_redundant=0,
				 n_repeated=0,
				 n_classes=2,
				 n_clusters_per_class=2
				 )

con_data = np.hstack((dataset[0],dataset[1].reshape(NMR_OF_SAMPLES,1)))

print("")
print(con_data)
print("")
print("shape and type")
print(con_data.shape)
print(con_data.dtype)
print("")

#f = open('linear_dataset_regression_1000000.txt','w')
f = open('linear_dataset_classification_1000000.txt','w')

for i in range(NMR_OF_SAMPLES):
	data=str(i+1)+" "
	for j in range(NMR_OF_FEAT):
		data+=str(dataset[0][i][j])+" "
	data+=str(dataset[1][i])
	f.write(data);
	f.write("\n");

f.close()