#include <iostream>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <random>
#include <map>
#include <chrono>
#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include "dsource.hh"
#include "H5Cpp.h"

using std::cout;
using std::endl;
using std::vector;
using namespace data_src;
using namespace H5;
using namespace mlpack;
using namespace arma;

#define NUM_OF_FEATS 100 // Number of features of each data point.
#define NUM_OF_SMPLS 50000// The total number of data points.

const int DSET_BUFF_SZ = 5000; // The number of data points the buffer contains.
const int DATA_SIZE = NUM_OF_FEATS + 2; // The size of each datapoint are the features plus id and label.
const int RANK = 2; // Number of dimensions of dataspace.
const H5std_string FILE_NAME("TestFile2.h5"); // Name of hdf5 file.
const H5std_string DATASET_NAME("Linear"); // Name of dataset.

typedef std::chrono::time_point<std::chrono::steady_clock> chr_time;

void Load_Arma_Data_to_H5(string filename,string datasetName,string numberOfPoints,bool append);

int main(void){
	
	cout << endl;
	
	double buffered_dataset[DSET_BUFF_SZ*DATA_SIZE]; // A buffered dataset to writen in hdf5 file
	mat data(&buffered_dataset[0], DATA_SIZE, DSET_BUFF_SZ, false, false);

	try{
		
		// Open the file and the dataset.
		H5File* file = new H5File(FILE_NAME, H5F_ACC_RDONLY);
	    DataSet* dataset = new DataSet (file->openDataSet( DATASET_NAME ));
		
		/*
	     *     Subset attributes.
	     */
	    hsize_t offset[2], countt[2];
	    hsize_t dimsm[2];

		offset[0] = 0; offset[1] = 0; // The offset of the starting element of the specified hyperslab.
	    countt[0] = DSET_BUFF_SZ; countt[1] = DATA_SIZE;// The number of elements along that dimension.
	    dimsm[0] = DSET_BUFF_SZ; dimsm[1] = DATA_SIZE;// Size of selected subset of dataset.

		// Get dataspace of the dataset.
		DataSpace dspace = dataset->getSpace();
		dspace.selectHyperslab(H5S_SELECT_SET, countt, offset);

		// Create memory dataspace.
		offset[0] = 0;
		dimsm[0] = DSET_BUFF_SZ;
		DataSpace mem2(RANK,dimsm);
		mem2.selectAll();

		dataset->read(buffered_dataset, PredType::NATIVE_DOUBLE, mem2, dspace);
		
		delete dataset;
		delete file;
		
		cout<<data<<endl;
		
	}// end of try block
	// catch failure caused by the H5File operations
   	catch( FileIException error ){
  		error.printError();
      	return -1;
   	}
    // catch failure caused by the DataSet operations
   	catch( DataSetIException error ){
  		error.printError();
      	return -1;
   	}
   	// catch failure caused by the DataSpace operations
   	catch( DataSpaceIException error ){
  		error.printError();
      	return -1;
   	}
   	// catch failure caused by the DataSpace operations
   	catch( DataTypeIException error ){
  		error.printError();
      	return -1;
   	}
	
	return 0;
}

void Load_Arma_Data_to_H5(string filename,string datasetName,string numberOfPoints,bool append){
	
	mat data;
	string inputFile = "linear_dataset"+numberOfPoints+".txt";
	
	data::Load(inputFile,data);
	if(!append){
		data.save(hdf5_name(filename,datasetName),hdf5_binary);
	}else{
		data.save(hdf5_name(filename,datasetName,hdf5_opts::append),hdf5_binary);
	}
	
	cout<<endl<<"Dataset "<< datasetName << "has been saved to " << filename << endl;
	
}