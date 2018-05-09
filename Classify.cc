#include <iostream>
#include <vector>

#include "Classifiers.hh"

using namespace ML_Classification;
using namespace data_src;

const string cfg = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/inputs.json";

int main(void){
	
	//Simple_Classification< PassiveAgressiveClassifier, double >* work;
	Simple_Classification< KernelPassiveAgressiveClassifier, double >* work;
	
	try{
		//work = new Simple_Classification< PassiveAgressiveClassifier, double >(cfg);
		work = new Simple_Classification< KernelPassiveAgressiveClassifier, double >(cfg);
	}catch(...){
		cout << "Terminating the execution." << endl;
		return 0;
	}
	
	// Train the classifier.
	work->Train();
	
	// Evaluate the classifier.
	work->getScore( work->getTestSet() , work->getTestSetLabels() );
	
	return 0;
}