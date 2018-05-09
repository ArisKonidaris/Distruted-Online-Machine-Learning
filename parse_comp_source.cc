#include <iostream>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <random>
#include <map>
#include <chrono>
#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include "data_structures.hh"
#include "dsource.hh"
#include "H5Cpp.h"

using std::cout;
using std::endl;
using std::vector;
using data_str::data_pnt_ld;
using namespace data_src;
using namespace H5;

#define NUM_OF_FEATS 100 // Number of features of each data point.
#define DATA_SIZE NUM_OF_FEATS+2 // Number of features of each data point.
#define DSET_BUFF_SZ 5000 // The number of data points the buffer contains.
#define NUM_OF_SMPLS 50000// The number of data points.

const H5std_string FILE_NAME("TestFile1.h5"); // Name of hdf5 file.
const H5std_string DATASET_NAME("Linear50000"); // Name of dataset.

typedef std::chrono::time_point<std::chrono::steady_clock> chr_time;

int main(void){
	
	cout << endl;
	
	int count=0;
	hdf5CompoundSource<data_pnt_ld> dt_src(FILE_NAME, DATASET_NAME, NUM_OF_FEATS, NUM_OF_SMPLS);

	while(dt_src.isValid()){

		cout<<dt_src.get().id<<" ";
		for(int j=0;j<NUM_OF_FEATS;j++){
			cout<<dt_src.get().features[j]<<" ";
		}
		cout << dt_src.get().label << endl << endl;
		count++;
		dt_src.advance();
	}

	cout<<"count : "<<count<<endl<<endl;
	
	return 0;
}