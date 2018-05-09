#include <iostream>
#include <string>
#include <vector>

#include "Regressors.hh"

using std::string;
using namespace data_src;
using namespace ML_Regression;

const string cfg = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/inputs.json";

int main(void){
	
	// Simple Regression job.
	Simple_Regression< PassiveAgressiveRegression, double >* work;
	
	try{
		work = new Simple_Regression< PassiveAgressiveRegression, double >(cfg);
	}catch(...){
		cout << "Terminating the execution." << endl;
		return 0;
	}
	
	// Train the classifier.
	work->Train();
	
	// Evaluate the classifier.
	double& RMSE = work->getRMSE(work->getTestSet(), work->getTestSetLabels());
	work->printPredictions(work->getTestSet(), work->getTestSetLabels());
	
	return 0;
	
}