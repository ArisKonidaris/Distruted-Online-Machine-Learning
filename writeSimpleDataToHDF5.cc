#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <typeinfo>
#include <chrono>
#include "data_structures.hh"
#include "H5Cpp.h"

using std::cout;
using std::endl;
using std::string;
using data_str::printAlgorithmInfo;
using std::min;
using namespace H5;

//#define NUM_OF_FEATS 100 // Number of features of each data point.
#define NUM_OF_FEATS 784 // Number of features of each data point.
//#define NUM_OF_FEATS 928 // Number of features of each data point.
//#define NUM_OF_FEATS 18 // Number of features of each data point.
//#define NUM_OF_SMPLS 50 // The number of data points.
//#define NUM_OF_SMPLS 5000 // The number of data points.
#define NUM_OF_SMPLS 10000 // The number of data points.
//#define NUM_OF_SMPLS 15000 // The number of data points.
//#define NUM_OF_SMPLS 20000 // The number of data points.
//#define NUM_OF_SMPLS 60000 // The number of data points.
//#define NUM_OF_SMPLS 190000 // The number of data points.
//#define NUM_OF_SMPLS 600000 // The number of data points.
//#define NUM_OF_SMPLS 1000000 // The number of data points.
//#define NUM_OF_SMPLS 2200000 // The number of data points.
//#define NUM_OF_SMPLS 2220000 // The number of data points.
#define DSET_BUFF_SZ 1000 // The number of data points the buffer contains.

const int BUFFER_SIZE(4<<20); // The size (in bytes) of the file stream buffer.

const int RANK = 2; // Number of dimensions of dataspace.
const int DATA_SIZE = NUM_OF_FEATS + 2;
const H5std_string FILE_NAME("Experimental_Datasets.h5"); // Name of hdf5 file.
//const H5std_string DATASET_NAME("Linear50"); // Name of dataset.
//const H5std_string DATASET_NAME("Linear5000"); // Name of dataset.
//const H5std_string DATASET_NAME("Linear600000"); // Name of dataset.
//const H5std_string DATASET_NAME("Linear1000000"); // Name of dataset.
//const H5std_string DATASET_NAME("MNIST_train"); // Name of dataset.
//const H5std_string DATASET_NAME("AMNIST"); // Name of dataset.
//const H5std_string DATASET_NAME("NAMNIST"); // Name of dataset.
//const H5std_string DATASET_NAME("C1_Train"); // Name of dataset.
//const H5std_string DATASET_NAME("C2_Train"); // Name of dataset.
//const H5std_string DATASET_NAME("C1C2_Train"); // Name of dataset.
//const H5std_string DATASET_NAME("C3_Train"); // Name of dataset.
//const H5std_string DATASET_NAME("C4_Train"); // Name of dataset.
//const H5std_string DATASET_NAME("C5_Train"); // Name of dataset.
//const H5std_string DATASET_NAME("MNIST_Test"); // Name of dataset.
//const H5std_string DATASET_NAME("NMNIST_Test"); // Name of dataset.
//const H5std_string DATASET_NAME("C1_Test"); // Name of dataset.
const H5std_string DATASET_NAME("C2_Test"); // Name of dataset.
//const H5std_string DATASET_NAME("C1C2_Test"); // Name of dataset.
//const H5std_string DATASET_NAME("C3_Test"); // Name of dataset.
//const H5std_string DATASET_NAME("C4_Test"); // Name of dataset.
//const H5std_string DATASET_NAME("C5_Test"); // Name of dataset.
//const H5std_string DATASET_NAME("LinearReg1000000"); // Name of dataset.


typedef std::chrono::time_point<std::chrono::steady_clock> chr_time;


int main(void){

	int count = 0; // A simple counter.
	int row = 0; // A simple buffer counter.
	int col = 0; // A simple buffer counter.
	int rd = 0; // Number of elements read from file it's round;
	double buffered_dataset[DSET_BUFF_SZ*DATA_SIZE]; // A buffered dataset to writen in hdf5 file
	std::ifstream myfile; // File for reading.
	std::fstream::streampos first, last; // Position tracker in the file.
	chr_time start; // Starting time of the algorithm.
	chr_time end; // Ending time of the algorithm.
	string line; // A character array for reading lines/data points.
	char* mybuffer = new char[BUFFER_SIZE]; // Buffer of size 4MBytes.

	try{
		
		// Create the file.
		H5File* file;
		try{
			file = new H5File(FILE_NAME, H5F_ACC_RDWR);
		}catch(...){
			file = new H5File(FILE_NAME, H5F_ACC_TRUNC);
		}

		// Create the data space.
	  	hsize_t dim[] = {NUM_OF_SMPLS,DATA_SIZE}; /* Dataspace dimensions */
		DataSpace dspace(RANK, dim);

	    // Create property list for a dataset and set up fill values.
	    double initial_point;
	    DSetCreatPropList plist;
	    plist.setFillValue(PredType::NATIVE_DOUBLE, &initial_point);

	    // Create the dataset.
	    DataSet* dataset;
	    dataset = new DataSet(file->createDataSet(DATASET_NAME, PredType::NATIVE_DOUBLE, dspace, plist));

	    // Subset attributes.
	    hsize_t offset[2], countt[2];
	    hsize_t dimsm[2];

	    offset[0] = -DSET_BUFF_SZ; offset[1] = 0; // The offset of the starting element of the specified hyperslab.
	    countt[0] = DSET_BUFF_SZ; countt[1] = DATA_SIZE;// The number of elements along that dimension.
	    dimsm[0] = DSET_BUFF_SZ; dimsm[1] = DATA_SIZE;// Size of selected subset of dataset.

	    myfile.rdbuf()->pubsetbuf(mybuffer, BUFFER_SIZE);
		//myfile.open("linear_dataset50.txt"); // Opening dataset file.
		//myfile.open("linear_dataset5000.txt"); // Opening dataset file.
		//myfile.open("linear_dataset600000.txt"); // Opening dataset file.
		//myfile.open("linear_dataset1000000.txt"); // Opening dataset file.
		//myfile.open("MNIST_train.txt"); // Opening dataset file.
		//myfile.open("AMNIST.txt"); // Opening dataset file.
		//myfile.open("NAMNIST.txt"); // Opening dataset file.
		//myfile.open("C1_Train.txt"); // Opening dataset file.
		//myfile.open("C2_Train.txt"); // Opening dataset file.
		//myfile.open("C1C2_Train.txt"); // Opening dataset file.
		//myfile.open("C3_Train.txt"); // Opening dataset file.
		//myfile.open("C4_Train.txt"); // Opening dataset file.
		//myfile.open("C5_Train.txt"); // Opening dataset file.
		//myfile.open("MNIST_test.txt"); // Opening dataset file.
		//myfile.open("NMNIST_test.txt"); // Opening dataset file.
		//myfile.open("C1_Test.txt"); // Opening dataset file.
		myfile.open("C2_Test.txt"); // Opening dataset file.
		//myfile.open("C1C2_Test.txt"); // Opening dataset file.
		//myfile.open("C3_Test.txt"); // Opening dataset file.
		//myfile.open("C4_Test.txt"); // Opening dataset file.
		//myfile.open("C5_Test.txt"); // Opening dataset file.
		//myfile.open("Stream_MNIST_test.txt"); // Opening dataset file.
		//myfile.open("linear_dataset_Regression_1000000.txt"); // Opening dataset file.
		//myfile.open("SUSY_Dataset_Normalized.txt"); // Opening dataset file.
		//myfile.open("SUSY_Dataset.txt"); // Opening dataset file.
		
		if(myfile.is_open()){
		
			cout << endl
				 << "File is open" 
				 << endl;

			// Calculating the size of the file.
			first = myfile.tellg(); // Get the beginning position in the file.
			myfile.seekg(0,std::ios::end); // Move to the last position of the file.
			last = myfile.tellg(); // Get the ending position in the file.
			myfile.seekg(0,std::ios::beg); // Go back to the beginning of the file.

			// Read from file and write to new a hdf5 dataset.
			start = std::chrono::steady_clock::now(); // Start counting time.

			while(getline(myfile,line)){ // Read line by line.
				
				std::string parser; // A parser buffer.

				for(u_int i=0;i<line.length();i++){
					if (i==line.length()-1){
						parser+=line[i];
						buffered_dataset[row*DATA_SIZE + col]=strtod(parser.c_str(),NULL);
						col=0;
						break;
					}
					else if (line[i]!=' '){
						parser+=line[i];
					}
					else{
						buffered_dataset[row*DATA_SIZE + col]=strtod(parser.c_str(),NULL);
						col++;
						parser.clear();
					}	
				}

				row++;
				rd++;
				count++;

		 		// Write buffered dataset to hdf5 file.
				if(row==DSET_BUFF_SZ || count==NUM_OF_SMPLS){
					row=0;
					offset[0] += DSET_BUFF_SZ;
					countt[0] = min(rd,DSET_BUFF_SZ);
					dimsm[0] = min(rd,DSET_BUFF_SZ);
					dspace.selectHyperslab(H5S_SELECT_SET, countt, offset);
	
					DataSpace memspace(RANK, dimsm);
					memspace.selectAll();
					
					// For debugging
		       		cout << endl << "rd : " << rd<<endl;
		       		cout << "offset[0] : " << offset[0] << endl;
					cout << "offset[1] : " << offset[1] << endl;
		       		cout << "countt[0] : " << countt[0] << endl;
					cout << "countt[1] : " << countt[1] << endl;
		       		cout << "dimsm[0] : " << dimsm[0] << endl;
					cout << "dimsm[1] : " << dimsm[1] << endl << endl;

		       		// Write subset to the dataset.
		       		dataset->write(buffered_dataset, PredType::NATIVE_DOUBLE, memspace, dspace);
			      	rd=0;
				}

			}

	   		// Release resources
	  		delete dataset;
	  		delete file;

			end = std::chrono::steady_clock::now(); // Stop counting time.

			//  Print info of line by line algorithm
			printAlgorithmInfo(count,start,end,first,last,BUFFER_SIZE);

	        // Open the file and the dataset.
        	file = new H5File( FILE_NAME, H5F_ACC_RDONLY );
	      	dataset = new DataSet (file->openDataSet( DATASET_NAME ));

		    // Get dataspace of the dataset.
		    dspace = dataset->getSpace();
		    offset[0] = 0; //45000
		    countt[0] = DSET_BUFF_SZ;
		    dspace.selectHyperslab(H5S_SELECT_SET, countt, offset);

     		// Create memory dataspace.
     		offset[0] = 0;
		    countt[0] = DSET_BUFF_SZ;
		    dimsm[0] = DSET_BUFF_SZ;
		    DataSpace mem2(RANK,dimsm);
			mem2.selectAll();

	      	dataset->read(buffered_dataset, PredType::NATIVE_DOUBLE, mem2, dspace);

	        // Display the fields.
	      	//cout << endl;
			//for(int i=0;i<DATA_SIZE;i++){
			//	cout << std::setprecision(12) << buffered_dataset[4999*DATA_SIZE+i] << " ";
			//}

	   		// Release resources
	  		delete dataset;
	  		delete file;

		}

		myfile.close(); // Close the file.

		cout << endl 
	     << "the file is closed" 
	     << endl
	     << endl;

	} // end of try block
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

   	cout << endl 
	     << "The dataset was written succesfully to the hdf5 file." 
	     << endl
	     << endl;
	
	return 0;
}