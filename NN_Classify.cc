#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "Classifiers.hh"

using std::string;
using namespace data_src;
using namespace ML_Classification;

const string cfg = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/inputs.json";
//typedef std::chrono::time_point<std::chrono::steady_clock> chr_time;

int main(void){
	
	//start = std::chrono::steady_clock::now(); // Start counting time.
	//end = std::chrono::steady_clock::now(); // Start counting time.
	//cout << std::chrono::duration<double, std::milli>(end-start).count() << endl;
	//LeNet_Classification* work;
	Extreme_Classification<ELM_Classifier,double>* work;
	
	try{
		//work = new LeNet_Classification(cfg);
		work = new Extreme_Classification<ELM_Classifier,double>(cfg);
	}catch(...){
		cout << "Terminating the execution." << endl;
		return 0;
	}
	
	// Train the classifier.
	work->Train();
	
	//cout<<"Training ended."<<endl;
	
	// Evaluate the classifier.
	//work->getScore(work->getTestSet(), work->getTestSetLabels());
	//work->printPredictions(work->getTestSet(), work->getTestSetLabels());
	
	return 0;
}