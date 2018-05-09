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

#include <dlib/dnn.h>
#include <dlib/cuda/tensor_tools.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>

using std::cout;
using std::endl;
using std::vector;

const size_t NUM_OF_SITES = 6;
const size_t NUM_OF_SVS = 10;
const size_t NUM_OF_FEATURES = 5;

using namespace dlib;
typedef std::chrono::time_point<std::chrono::steady_clock> chr_time;

int main(void){
	
	srand(static_cast<unsigned>(time(0)));
	chr_time start; // Starting time of the algorithm.
	chr_time end; // Ending time of the algorithm.
	resizable_tensor a(1000000,2,1,1);
	resizable_tensor b(1000000,2,1,1);
	
	for(size_t i=0;i<a.size();i++){
		a.host()[i]=(float)i;
		b.host()[i]=(float)i+static_cast<float>(std::rand())/static_cast<float>(RAND_MAX);
	}
	
	// Read from file and write to new a hdf5 dataset.
	start = std::chrono::steady_clock::now(); // Start counting time.
	
	resizable_tensor subtr(1000000,2,1,1);
	for(size_t i=0;i<subtr.size();i++){
		subtr.host()[i]=a.host()[i]-b.host()[i];
	}	
	cout <<"Dot : " << dot(subtr,subtr) << endl;
	
	end = std::chrono::steady_clock::now(); // Stop counting time.
	cout << "Dlib Performance : " << std::chrono::duration<double, std::milli>(end-start).count() << endl;
	
	// Read from file and write to new a hdf5 dataset.
	start = std::chrono::steady_clock::now(); // Start counting time.
	
	resizable_tensor sub(b);
	sub*=-1.;
	dlib::cuda::add(1.,sub,1.,a);
	cout <<"Dot : " << dot(sub,sub) << endl;
	
	end = std::chrono::steady_clock::now(); // Stop counting time.
	
	cout << "Dlib Performance : " << std::chrono::duration<double, std::milli>(end-start).count() << endl;
	
	return 0;
}