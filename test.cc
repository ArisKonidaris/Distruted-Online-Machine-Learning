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
	
	cout << sizeof(float) << endl;
	cout << sizeof(float)*857738 << endl;
		
	return 0;
}