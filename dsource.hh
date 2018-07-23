#ifndef _DSOURCE_HH_
#define _DSOURCE_HH_

#include <iostream>
#include <string>
#include <vector>
#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>
#include <typeinfo>
#include "H5Cpp.h"

namespace data_src {

using std::cout;
using std::endl;
using std::vector;
using std::string;
using namespace H5;
	
/*
 * A main-memory store of stream records.
 */
template <class T>
class buffered_dataset : public std::vector<T>
{	
public:
	using std::vector<T>::vector;
	void load(T point);
};

// Load data to the buffer / data container.
template <class T>
void buffered_dataset<T>::load(T point){
	this->push_back(point);
}

/**
	A data source is an object providing the data of a stream.
	The API is very similar to an iterator. 

	Data sources should only be held by `std::shared_ptr` 
	inside other objects or in functions.
*/
template <class T>
class data_source{
protected:
	T record;
	bool isvalid;
public:

	/**
		Return the current valid record or throw an expection.
	  */
	inline const T& get() { return record; }

	inline bool isValid() { return isvalid; }

	/**
		Advance the data source to the next record.
	  */
	virtual void advance() { }

	/**
		Rewind the data source.
	  */
	virtual void rewind() { }

};


/*********************************************
			 hdf5CompoundSource               
*********************************************/	

template <class T>
struct hdf5CompoundSource : public data_source<T>{

	CompType record_type; // a derivative of a DataType that operates on HDF5 compound datatypes. 

	H5File* file;  // HDF5 file.
	DataSet* dataset; // HDF5 dataset.
	DataSpace dspace; // HDF5 dataspace.
	hsize_t dataset_length; // Total number of training examples.

	buffered_dataset<T> buffer; // the buffer.
	DataSpace mspace; // Memory dataspace.
	hsize_t buffer_size; // size of buffer.

	hsize_t curpos; // current possition in the dataset.
	typename buffered_dataset<T>::iterator currec;

	hdf5CompoundSource(H5std_string fname, H5std_string dname, hsize_t num_of_feats, hsize_t num_of_pts);

	// Rewinds the whole dataset.
	void rewind() override;

	// Resizes the buffer.
	void resize_buffer(hsize_t bsize);

	/**
		Advances the iterator of the buffer to the next record.
		*/
	void advance() override;

	/** 
		Method that reads a chunk of data from a HDF5 file
		and loads it to the buffer.
		*/
	void fill_buffer();
	
	/** 
		Returns the buffer containing the data.
		The buffer is returned by reference to avoid needless copying.
		*/
	inline buffered_dataset<T>& getbuffer() { return buffer; }

};

template <class T>
hdf5CompoundSource<T>::hdf5CompoundSource(H5std_string fname, H5std_string dname, hsize_t num_of_feats, hsize_t num_of_pts)
{
	dataset_length = num_of_pts;

	// Create the record type in memory.
	hsize_t array_dim[] = {num_of_feats};
	hid_t array_tid = H5Tarray_create(H5T_NATIVE_LDOUBLE, 1, array_dim);
	
	record_type = CompType(sizeof(T));
	record_type.insertMember("ID", 
		HOFFSET(T, id), PredType::NATIVE_INT );
	record_type.insertMember("FEATURES", 
		HOFFSET(T, features), array_tid );
	record_type.insertMember("LABEL", 
		HOFFSET(T, label), PredType::NATIVE_LDOUBLE );

	// Open the file and dataset.
	H5File* file = new H5File( fname, H5F_ACC_RDONLY );
	dataset = new DataSet (file->openDataSet( dname ));

	// Get dataspace of the dataset.
	dspace = dataset->getSpace();
	
	// start iteration
	rewind();

}

template <class T>
void hdf5CompoundSource<T>::rewind()
{
	// Prepare for iteration
	this->isvalid = true;

	// 5000 data points per read
	resize_buffer(5000);

	// position at start of dataset
	curpos = 0;

	advance();
}

template <class T>
void hdf5CompoundSource<T>::resize_buffer(hsize_t bsize)
{
	// update this for easy access
	buffer_size = bsize;

	// resize the buffer as required.
	// note: the above invalidates currec iterator, so reset it!
	buffer.resize(bsize);
	currec = buffer.end();

	// resize the mspace DataSpace to match new buffer
	mspace.setExtentSimple(1,&buffer_size);
	mspace.selectAll();
}

template <class T>
void hdf5CompoundSource<T>::advance()
{
	if(! this->isvalid) return;

	if(currec==buffer.end()) 
	{
		// try to read in next slab
		if(curpos == dataset_length) {
			this->isvalid = false;
			resize_buffer(0);
			return;
		}

		// let us read the next slab!
		if(dataset_length-curpos < buffer.size())
			resize_buffer(dataset_length-curpos);

		fill_buffer();
		currec = buffer.begin();
	}

	this->record=*currec;
	++currec;
}

template <class T>
void hdf5CompoundSource<T>::fill_buffer()
{
	// the file space
	dspace.selectHyperslab(H5S_SELECT_SET, &buffer_size, &curpos);

	// move data
	dataset->read(buffer.data(), record_type, mspace, dspace);

	// advance
	curpos += buffer_size;
}


/*********************************************
			     hdf5Source               
*********************************************/

template <class T>
struct hdf5Source : public data_source<T>, public boost::enable_shared_from_this<hdf5Source<T>>{

	H5File* file;  // HDF5 file.
	DataSet* dataset; // HDF5 dataset.
	DataSpace dspace; // HDF5 dataspace.
	hsize_t dataset_length; // Total number of training examples.
	hsize_t data_size; // Number of features plus id and label.

	std::vector<T> buffer; // The buffer.
	DataSpace mspace; // Memory dataspace.
	hsize_t buffer_size; // Size of buffer (in data points).

	hsize_t curpos; // current possition in the dataset.
	typename std::vector<T>::iterator currec;
	bool load_ids; // If false, the ids are not loaded from the disk.

	hdf5Source(H5std_string fname, H5std_string dname, bool ld_ids);

	// Rewinds the whole dataset.
	void rewind() override;
	
	// Resizes the buffer.
	void resize_buffer(hsize_t bsize);

	/**
		Advances the iterator of the buffer to the next chunk of data points.
		*/
	void advance() override;

	/** 
		Method that reads a chunk of data from a HDF5 file
		and loads it to the buffer.
		*/
	void fill_buffer();
	
	/** 
		Returns the buffer containing the data.
		The buffer is returned by reference to avoid needless copying.
		*/
	std::vector<T>& getbuffer() { return buffer; }
	
	inline int getBufferSize() const { return (int)buffer_size; }
	inline int getDatasetLength() const { return (int)dataset_length; }
	inline int getDataSize() const { return (int)data_size; }
	inline int getCurrentPos() const { return (int)curpos; }
	
	// Get a shared pointer from this object.
	boost::shared_ptr<hdf5Source<T>> shareSource(){
		return this->shared_from_this();
	}

};

template<typename T>
hdf5Source<T>::hdf5Source(H5std_string fname, H5std_string dname, bool ld_ids)
{
	
	// Open the file and dataset.
	H5File* file = new H5File( fname, H5F_ACC_RDONLY );
	dataset = new DataSet ( file->openDataSet( dname ) );

	// Get dataspace of the dataset.
	dspace = dataset->getSpace();
	
	hsize_t dims_out[2];
	dspace.getSimpleExtentDims(dims_out,NULL);
	
	// Declare the necessary object variables
	dataset_length = dims_out[0];
	load_ids = ld_ids;
	data_size = dims_out[1];
	if(!ld_ids){
		data_size -= 1;
	}
	
	cout << endl << "Dataset size : " << dataset_length << endl;
	
	// start iteration
	rewind();

}

template<typename T>
void hdf5Source<T>::rewind()
{
	// Prepare for iteration
	this->isvalid = true;

	// 1000 data points per read
	resize_buffer(1000);

	// position at start of dataset
	curpos = 0;

	advance();
}

template<typename T>
void hdf5Source<T>::resize_buffer(hsize_t bsize)
{
	// update this for easy access
	buffer_size = bsize;

	// resize the buffer as required.
	// note: the above invalidates currec iterator, so reset it!
	buffer.resize(bsize*data_size);
	currec = buffer.end();

	// resize the mspace DataSpace to match new buffer
	hsize_t dims[2] = {bsize,data_size};
	mspace.setExtentSimple(2,dims);
	mspace.selectAll();
}

template<typename T>
void hdf5Source<T>::advance()
{
	if(! this->isvalid) return;

	if(currec==buffer.end()) 
	{
		// try to read in next slab
		if(curpos == dataset_length) {
			this->isvalid = false;
			resize_buffer(0);
			return;
		}

		// let us read the next slab!
		if(dataset_length*data_size-curpos*data_size < buffer.size())
			resize_buffer( (dataset_length*data_size-curpos*data_size) / data_size );

		fill_buffer();
		currec = buffer.begin();
	}

	this->record=*currec;
	currec=buffer.end();
}

template<typename T>
void hdf5Source<T>::fill_buffer()
{
	// the file space
	hsize_t dims[2] = {buffer_size,data_size};
	hsize_t offset[2] = {curpos,0};
	offset[0] = curpos;
	if(load_ids){
		offset[1] = 0;
	}else{
		offset[1] = 1;
	}
	dspace.selectHyperslab(H5S_SELECT_SET, dims, offset);

	// move data
	dataset->read(buffer.data(), PredType::NATIVE_DOUBLE, mspace, dspace);
	// advance
	curpos += buffer_size;
}	

template<typename T>
boost::shared_ptr<hdf5Source<T>> getPSource(H5std_string fname, H5std_string dname, bool ld_ids){
	auto source = new hdf5Source<T>(fname, dname, ld_ids);
	return boost::shared_ptr<hdf5Source<T>>(source);
}

} // End of namespace data_src.
#endif