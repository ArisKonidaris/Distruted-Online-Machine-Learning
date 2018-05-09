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
using data_str::data_pnt_ld;
using data_str::space_parsing;
using data_str::printAlgorithmInfo;
using std::min;
using namespace H5;

#define NUM_OF_FEATS 100 // Number of features of each data point.
#define NUM_OF_SMPLS 50000 // The number of data points.
#define DSET_BUFF_SZ 5000 // The number of data points the buffer contains.

const int BUFFER_SIZE(4<<20); // The size (in bytes) of the file stream buffer.

const int RANK = 1; // Number of dimensions of dataspace.
const H5std_string FILE_NAME("TestFile1.h5"); // Name of hdf5 file.
const H5std_string DATASET_NAME("Linear50000"); // Name of dataset.

typedef std::chrono::time_point<std::chrono::steady_clock> chr_time;


int main(void){

	int count = 0; // A simple counter.
	int buf_cntr = 0; // A simple buffer counter.
	int rd = 0; // Number of elements read from file it's round;
	data_pnt_ld buffered_dataset[DSET_BUFF_SZ]; // A buffered dataset to writen in hdf5 file
	std::ifstream myfile; // File for reading.
	std::fstream::streampos first, last; // Position tracker in the file.
	chr_time start; // Starting time of the algorithm.
	chr_time end; // Ending time of the algorithm.
	string line; // A character array for reading lines/data points.
	char* mybuffer = new char[BUFFER_SIZE]; // Buffer of size 4MBytes.

	try{
		
		// Create the file.
		H5File* file = new H5File(FILE_NAME, H5F_ACC_TRUNC);

		// Create the data space.
	  	hsize_t dim[] = {NUM_OF_SMPLS}; /* Dataspace dimensions */
		DataSpace dspace(RANK, dim);

		// Create the array data type.
	    hsize_t array_dim[] = {NUM_OF_FEATS};
	    hid_t array_tid = H5Tarray_create(H5T_NATIVE_LDOUBLE, 1, array_dim);

		// Create the memory datatype.
	    CompType mtype( sizeof(data_pnt_ld) );
	    mtype.insertMember("ID", HOFFSET(data_pnt_ld, id), PredType::NATIVE_INT);
		mtype.insertMember("FEATURES", HOFFSET(data_pnt_ld, features), array_tid);
	    mtype.insertMember("LABEL", HOFFSET(data_pnt_ld, label), PredType::NATIVE_LDOUBLE);

	    // Create property list for a dataset and set up fill values.
	    data_pnt_ld initial_point;
	    DSetCreatPropList plist;
	    plist.setFillValue(mtype, &initial_point);

	    // Create the dataset.
	    DataSet* dataset;
	    dataset = new DataSet(file->createDataSet(DATASET_NAME, mtype, dspace, plist));

	    // Subset attributes.
	    hsize_t offset[1], countt[1];
	    hsize_t dimsm[1];

	    offset[0] = -DSET_BUFF_SZ; // The offset of the starting element of the specified hyperslab.
	    countt[0] = DSET_BUFF_SZ; // The number of elements along that dimension.
	    dimsm[0] = DSET_BUFF_SZ; // Size of selected subset of dataset.

	    myfile.rdbuf()->pubsetbuf(mybuffer,BUFFER_SIZE);
		myfile.open("linear_dataset50000.txt"); // Opening dataset file.
		
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

				space_parsing<data_pnt_ld> pr(line,buffered_dataset,buf_cntr,line.length());
				pr.parseLine();

				buf_cntr++;
				rd++;
				count++;

		 		// Write buffered dataset to hdf5 file.
				if(buf_cntr==DSET_BUFF_SZ || count==NUM_OF_SMPLS){
					buf_cntr=0;
					offset[0] += DSET_BUFF_SZ;
					countt[0] = min(rd,DSET_BUFF_SZ);
					dimsm[0] = min(rd,DSET_BUFF_SZ);
					dspace.selectHyperslab(H5S_SELECT_SET, countt, offset);

					DataSpace memspace(RANK, dimsm);
					memspace.selectAll();
					
					// For debugging
		       		cout<<endl<<"rd : "<<rd<<endl;
		       		cout<<"offset[0] : "<<offset[0]<<endl;
		       		cout<<"countt[0] : "<<countt[0]<<endl;
		       		cout<<"dimsm[0] : "<<dimsm[0]<<endl<<endl;

		       		// Write subset to the dataset.
		       		dataset->write(buffered_dataset, mtype, memspace, dspace);
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
		    offset[0] = 45000;
		    countt[0] = DSET_BUFF_SZ;
		    dspace.selectHyperslab(H5S_SELECT_SET, countt, offset);

     		// Create memory dataspace.
     		offset[0] = 0;
		    countt[0] = DSET_BUFF_SZ;
		    dimsm[0] = DSET_BUFF_SZ;
		    DataSpace mem2(RANK,dimsm);
     		mem2.selectHyperslab(H5S_SELECT_SET, countt, offset);

	      	dataset->read(buffered_dataset, mtype, mem2, dspace);

	        // Display the fields.
	      	cout << endl << buffered_dataset[4999].id << " ";
			for(int i=0;i<NUM_OF_FEATS;i++){
				cout<<std::setprecision(12)<< buffered_dataset[4999].features[i]<<" ";
			}
			cout<<std::setprecision(12)<<buffered_dataset[4999].label<<endl<<endl;

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