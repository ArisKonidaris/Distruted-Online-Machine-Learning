#ifndef _DATA_STRUCTURES_HH_
#define _DATA_STRUCTURES_HH_

#include <iostream>
#include <string>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <vector>

#define NUM_OF_FEATS_DSET 100

namespace data_str {

typedef std::chrono::time_point<std::chrono::steady_clock> chr;
typedef std::fstream::streampos pos;
using std::cout;
using std::endl;
using std::vector;

/*
 *     A data point with long double features for classification.
 *     Each data point has an id, x features, and a label.
 */
struct data_pnt_ld{
	int id;
	long double features[NUM_OF_FEATS_DSET];
	long double label;

	inline void print_point() const;

	inline void point_info() const;
};

inline void data_pnt_ld::print_point() const{
	cout<<endl<<"Data point"<<endl;
	cout<<"id : "<<id<<endl;
	for(int i=0;i<NUM_OF_FEATS_DSET;i++){
		cout << "feature_"+std::to_string(i)+" : " << features[i] << endl;
	}
	cout << "label : " << label << endl;
}

inline void data_pnt_ld::point_info() const
{
	cout << endl << "This is a data point with long double features for classification." << endl;
}


/*********************************************
				  parsing
*********************************************/	

template <class T>
class parsing{
protected:
	const std::string& line;
	T* buffered_dataset;
	const int counter;
	const int length;
public:
	parsing(const std::string& ln, T* buf_dt, const int cnt, const int len):
	line(ln),
	buffered_dataset(buf_dt),
	counter(cnt),
	length(len) { }
	
	auto getAddrLine() const { return &line; }
	
	auto getAddrBufDt() const { return buffered_dataset; }
	
	virtual void parseLine() { }
};


/*********************************************
			   space_parsing
*********************************************/	

template <class T>
class space_parsing : public parsing<T>{
public:
	space_parsing(const std::string& ln, T* buf_dt, const int cnt, const int len)
	:parsing<T>(ln, buf_dt, cnt, len) { }
	
	void parseLine() override;
};

template <class T>
void space_parsing<T>::parseLine(){
	std::string parser; // A parser buffer.
	int parse=-1; // A parser variable.

	for(int i=0;i<this->length;i++){
		if (i==this->length-1){
			parser+=this->line[i];
			this->buffered_dataset[this->counter].label=strtold(parser.c_str(),NULL);
			break;
		}
		else if (this->line[i]!=' '){
			parser+=this->line[i];
		}
		else{
			if(parse==-1){
				this->buffered_dataset[this->counter].id=atoi(parser.c_str());
				parse++;
			}
			else{
				this->buffered_dataset[this->counter].features[parse]=strtold(parser.c_str(),NULL);
				parse++;
			}
			parser.clear();
		}	
	}
}


/*********************************************
		 printAlgorithmInfo Function
*********************************************/	

void printAlgorithmInfo(int count, chr start, chr end, pos first, pos last, int BUFFER_SIZE){
	cout << endl
	<< "algorithm : line by line read"
	<< endl
	<< "count : "
	<< count
	<< endl
	<< "performance : "
	<< std::chrono::duration<double, std::milli>(end-start).count()
	<< " ms"
	<< endl
	<< "file size : "
	<< (float)(last-first)/(1024*1024)
	<< " Mbytes"
	<< " ("
	<< (last-first)
	<< " bytes)"
	<< endl
	<< "buffer size : "
	<< BUFFER_SIZE
	<< endl;
}

} // End of namespace data_str.
#endif