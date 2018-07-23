#include <iostream>
#include <string>
#include <vector>
#include <string>
#include <armadillo>
#include <set>
#include <chrono>
#include <ctime>
#include <cassert>
#include <iomanip>
#include <unordered_map>
#include <fstream>
#include <random>

#include <dlib/dnn.h>
#include <dlib/dnn/tensor_tools.h>
#include <dlib/cuda/tensor_tools.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>

#include "dsource.hh"
#include "dsarch.hh"

using std::cout;
using std::endl;
using std::vector;
using std::set;

const size_t NUM_OF_SITES = 6;
const size_t NUM_OF_SVS = 10;
const size_t NUM_OF_FEATURES = 5;

//using namespace dlib;
using namespace H5;
using namespace data_src;
typedef std::chrono::time_point<std::chrono::steady_clock> chr_time;
typedef boost::shared_ptr<hdf5Source<double>> PointToSource;

int main(void){
	
	long int seed = -1;
	if(seed>=0){
		std::srand (seed);
	}else{
		std::srand (time(&seed));
	}
	
	
	float percentage = 0.5;
	size_t nodes = 8;
	set<set<set<size_t>>> dist;
	
	set<size_t> B;
	set<size_t> B_compl;
	
	for(size_t i=0; i<nodes; i++){
		B_compl.insert(i);
	}
	
	for(size_t i=0; i<std::floor(nodes*percentage); i++){
		size_t n = std::rand()%(nodes);
		while(B.find(n) != B.end()){
			n = std::rand()%(nodes);
		}
		B.insert(n);
		B_compl.erase(n);
	}
	
	for(size_t n : B)
		cout << n << endl;
	cout << endl;
	for(size_t n : B_compl)
		cout << n << endl;
		
	cout << "//////" << endl;
	
	int p[nodes]={};
	for(size_t i =0; i<100; i++){
		double n = ((double) rand() / (RAND_MAX));
		if(n<0.75){
			auto it = B.begin();
			std::advance(it, (int)(std::rand()%(B.size())));
			++p[*it];
		}else{
			auto it = B_compl.begin();
			std::advance(it, (int)(std::rand()%(B_compl.size())));
			++p[*it];
		}
	}
		
	for (size_t i=0; i<nodes; ++i)
		cout << i << " : " << p[i] << endl;	
		
	return 0;
}