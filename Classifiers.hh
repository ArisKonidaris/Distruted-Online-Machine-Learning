#ifndef _CLASSIFIERS_HH_
#define _CLASSIFIERS_HH_

#include <iostream>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <random>
#include <algorithm>
#include <jsoncpp/json/json.h>
#include <cmath>
#include <random>
#include <time.h>
#include <fstream>

#include <mlpack/core.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/core/optimizers/ada_grad/ada_grad.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/core/optimizers/adam/adam.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/network_init.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/init_rules/kathirvalavakumar_subavathi_init.hpp>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>

#include "dsource.hh"

namespace ML_Classification {

using std::cout;
using std::endl;
using std::vector;
using std::string;
using namespace data_src;
using namespace H5;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::math;
using namespace mlpack::optimization;
using namespace dlib;

/*********************************************
	Passive Aggressive Classifier
*********************************************/

class PassiveAgressiveClassifier{
protected:
	string name = "Passive Aggressive Classifier.";
	arma::dvec W; // Parameter vector.
	string regularization; // Regularization technique.
	double C; // Regularization C term. [optional]
	Json::Value root; // JSON file to read the hyperparameters.  [optional]
	
public:
	PassiveAgressiveClassifier(string cfg);
	
	// Stream update.
	inline void fit(arma::mat& batch, arma::dvec& labels);
	
	// Make a prediction.
	arma::dvec predict(arma::mat& batch);
	
	// Make a prediction.
	inline double predict(arma::dvec& data_point);
	
	// Get score.
	inline vector<double> accuracy(arma::mat& testbatch, arma::dvec& labels);
	
	// Information getters.
	inline string getName() { return name; }
	inline arma::dvec& getModel() { return W; }
	inline string getRegularization() { return regularization; }
	inline double getC() { (regularization=="none") ? 0.0 : C; }
	
};

PassiveAgressiveClassifier::PassiveAgressiveClassifier(string cfg){
	try{
		std::ifstream cfgfile(cfg);
		cfgfile >> root;
		regularization = root["parameters"]
						.get("regularization","Incorrect regularization parameter").asString();
		if(regularization=="Incorrect regularization parameter" ||
		   (regularization!="none" &&
			regularization!="l1" &&
			regularization!="l2")){
			cout << endl << "Incorrect regularization parameter." << endl;
			cout << "Acceptable regularization parameters : ['none','l1','l2']" << endl;
			throw;
		}
		if(regularization!="none"){
			C = root["parameters"].get("C",-1).asDouble();
			if(C<0){
				cout << endl << "Incorrect parameter C." << endl;
				cout << endl << "Acceptable C parameters : [positive double]" << endl;
				throw;
			}
		}
	}catch(...){
		throw;
	}
}

inline void PassiveAgressiveClassifier::fit(arma::mat& batch, arma::dvec& labels){
		
	// Initialize the parameter vector W if has no yet been initialized.
	if (W.n_rows==0){
		W = arma::zeros<arma::dvec>(batch.n_rows);
	}
	
	// Starting the online learning
	for(int i=0; i<batch.n_cols; i++){
		
		// calculate the Hinge loss.
		double loss = 1. - labels(i)*arma::dot(W,batch.unsafe_col(i));
		
		// Classified correctly. Parameters stay the same. Passive approach.
		if (loss <= 0.){
			continue;
		}
			
		// Calculate the Lagrange Multiplier.
		double Lagrange_Muliplier;
		if(regularization=="none"){
			Lagrange_Muliplier = loss / arma::dot(batch.unsafe_col(i),batch.unsafe_col(i)) ;
		}else if (regularization=="l1"){
			Lagrange_Muliplier = std::min(C,loss / arma::dot(batch.unsafe_col(i),batch.unsafe_col(i)));
		}else if (regularization=="l2"){
			Lagrange_Muliplier = loss / ( arma::dot(batch.unsafe_col(i),batch.unsafe_col(i)) + 1/(2*C) );
		}else{
			//throw exception
			cout << "throw exception" << endl;
		}
		
		// Update the parameters.
		W += Lagrange_Muliplier*labels(i)*batch.unsafe_col(i);
	}
	
}

arma::dvec PassiveAgressiveClassifier::predict(arma::mat& batch){
	arma::dvec prediction = arma::zeros<arma::dvec>(batch.n_cols);
	for(int i=0;i<batch.n_cols;i++){
		if( arma::dot(W,batch.unsafe_col(i)) >= 0. ){
			prediction(i) = 1.;
		}else{
			prediction(i) = -1.;
		}
	}
	return prediction;
}

inline double PassiveAgressiveClassifier::predict(arma::dvec& data_point){
	double prediction;
	if( arma::dot(W,data_point) >= 0. ){
		prediction = 1.;
	}else{
		prediction = -1.;
	}
	return prediction;
}

inline vector<double> PassiveAgressiveClassifier::accuracy(arma::mat& testbatch, arma::dvec& labels){
	
	// Calculate accuracy.
	int errors=0;
	for(int i=0;i<labels.n_elem;i++){
		double prediction;
		if( arma::dot(W,testbatch.unsafe_col(i)) >= 0. ){
			prediction = 1.;
		}else{
			prediction = -1.;
		}
		if (labels(i)!=prediction){
			errors++;
		}
	}
	
	vector<double> score;
	score.push_back(100.0*(labels.n_elem-errors)/(labels.n_elem));
	score.push_back((double)errors);
	
	return score;
	
}


/*********************************************
	Kernel Passive Aggressive Classifier
*********************************************/

class Kernel{
public:
	Kernel() { }
	virtual inline double HilbertDot(arma::dvec& a, arma::dvec& b) { };
};

class PolynomialKernel : public Kernel{
protected:
	int degree;
	double offset;
public:
	PolynomialKernel(int deg, double off):degree(deg), offset(off) { }
	PolynomialKernel(int deg):degree(deg), offset(0.) { }
	PolynomialKernel(double off):degree(2), offset(off) { }
	PolynomialKernel():degree(2), offset(0.) { }
	
	// Setters.
	inline void setDegree(int deg) { degree=deg; }
	inline void setOffset(double off) { offset=off; }
	
	// Getters.
	inline int getDegree() { return degree; }
	inline double getOffset() { return offset; }
	
	inline double HilbertDot(arma::dvec& a, arma::dvec& b) override
	{
		return std::pow(arma::dot(a,b) + offset, degree);
	};
	
};

class RbfKernel : public Kernel{
protected:
	double sigma;
public:
	RbfKernel(double sig):sigma(sig) { }
	RbfKernel():sigma(1.) { }
	
	// Setters.
	inline void setSigma(double sig) { sigma=sig; }
	
	// Getters.
	inline double getSigma() { return sigma; }
	
	inline double HilbertDot(arma::dvec& a, arma::dvec& b) override
	{
		arma::dvec c = a-b;
		return std::exp( -( arma::dot(c,c) ) / ( 2*std::pow(sigma,2) ) );
	}
	
};


/*********************************************
	Kernel Passive Aggressive Classifier
*********************************************/

class KernelPassiveAgressiveClassifier{
protected:
	string name = "Passive Aggressive Classifier with Kernel.";
	vector<arma::dvec> SVs; // Support vectors.
	vector<double> coef; // The coefficients of the support vectors.
	string regularization; // Regularization technique.
	double C; // Regularization C term. [optional]
	Json::Value root; // JSON file to read the hyperparameters.  [optional]
	Kernel* kernel; // The Kernel to be used by the classifier.
	int maxSVs; // Maximum number of support vectors to be held in memory.
	
	double a; // Sparse Passive Aggressive parameter a.
	double b; // Sparse Passive Aggressive parameter b.
	std::default_random_engine generator;
public:
	KernelPassiveAgressiveClassifier(string cfg);
	
	// Stream update.
	inline void fit(arma::mat& batch, arma::dvec& labels);
	
	// Make a prediction.
	arma::dvec predict(arma::mat& batch);
	
	// Make a prediction.
	inline double predict(arma::dvec& data_point);
	
	// Get the margin.
	inline double Margin(arma::dvec& data_point);
	
	// Get score.
	inline vector<double> accuracy(arma::mat& testbatch, arma::dvec& labels);
	
	// Information getters.
	inline string getName() { return name; }
	inline vector<arma::dvec>& getSupportVectors() { return SVs; }
	inline void eraseSV(int i) { SVs.erase(SVs.begin()+i); }
	inline vector<double>& getCoefficients() { return coef; }
	inline string getRegularization() { return regularization; }
	inline double getC() { (regularization=="none") ? 0.0 : C; }
	inline Kernel* getKernel() { return kernel; }
	inline int getNumberOfSVs() { return SVs.size(); }
	
};

KernelPassiveAgressiveClassifier::KernelPassiveAgressiveClassifier(string cfg){
	try{
		
		std::ifstream cfgfile(cfg);
		cfgfile >> root;
		
		string ker = root["parameters"]
					 .get("kernel","No Kernel given").asString();
		if(ker!="poly" && ker!="rbf"){
			cout << endl << "Invalid kernel given." << endl;
			cout << "Valid kernels [\"rbf\",\"poly\"] ." << endl;
			throw;
		}
		if(ker=="rbf"){
			double sigma = root["parameters"].get("sigma",-1).asDouble();
			if(sigma<=0.){
				cout << endl << "Invalid sigma given." << endl;
				cout << "Sigma must be a positive double." << endl;
				cout << "Sigma is set to 1.0 by default" << endl;
				kernel = new RbfKernel();
			}else{
				kernel = new RbfKernel(sigma);
			}
		}else if(ker=="poly"){
			int degree = root["parameters"].get("degree",-1).asInt();
			double offset = root["parameters"].get("offset",1e-20).asDouble();
			
			if(degree<=0 && offset!=1e-20){
				cout << endl << "Invalid degree given." << endl;
				cout << "Degree must be a positive integer." << endl;
				cout << "Sigma is set to 2.0 by default" << endl;
				kernel = new PolynomialKernel(offset);
			}else if(degree>0 && offset==1e-20){
				cout << endl << "Offset id 0.0 by default" << endl;
				kernel = new PolynomialKernel(degree);
			}else if(degree<=0 && offset==1e-20){
				cout << endl << "Invalid degree given." << endl;
				cout << "Degree must be a positive integer." << endl;
				cout << "Sigma is set to 2.0 by default" << endl;
				cout << endl << "Offset id 0.0 by default" << endl;
				kernel = new PolynomialKernel();
			}else{
				kernel = new PolynomialKernel(degree,offset);
			}
			
		}
		
		regularization = root["parameters"]
						.get("regularization","Incorrect regularization parameter").asString();
		if(regularization=="Incorrect regularization parameter" ||
		   (regularization!="none" &&
			regularization!="l1" &&
			regularization!="l2")){
			cout << endl << "Incorrect regularization parameter." << endl;
			cout << "Acceptable regularization parameters : ['none','l1','l2']" << endl;
			throw;
		}
		if(regularization!="none"){
			C = root["parameters"].get("C",-1).asDouble();
			if(C<0){
				cout << endl << "Incorrect parameter C." << endl;
				cout << endl << "Acceptable C parameters : [positive double]" << endl;
				throw;
			}
		}
		
		double alpha = root["parameters"].get("a",-1.).asDouble();
		double beta = root["parameters"].get("b",-1.).asDouble();
		if(alpha<=0.){
			cout << endl << "Invalid parameter a." << endl;
			cout << "Parameter a must be a positive double." << endl;
			throw;
		}
		if(beta<=0.){
			cout << endl << "Invalid parameter b." << endl;
			cout << "Parameter b must be a positive double." << endl;
			throw;
		}
		if(alpha>beta){
			cout << endl << "Invalid parameters a and b.";
			cout << "Parameter b cannot be bigger that a." << endl;
			throw;
		}
		a=alpha;
		b=beta;
		
		int mSVs = root["parameters"].get("maxSVs",-1.).asInt64();
		if(mSVs<=0){
			cout << endl << "Invalid parameter maxSVs." << endl;
			cout << "The parameter maxSVs must be a positive integer." << endl;
			throw;
		}
		maxSVs = mSVs;
		
	}catch(...){
		throw;
	}
}

inline void KernelPassiveAgressiveClassifier::fit(arma::mat& batch, arma::dvec& labels){
	// Starting the online learning
	for(int i=0; i<batch.n_cols; i++){
		double loss;
		arma::dvec point = batch.unsafe_col(i);
		
		if(SVs.size()==0){
			loss = 1.;
		}else{
			// calculate the Hinge loss.
			loss = 1. - labels(i)*this->Margin(point);
		}
		
		// Classified correctly. Parameters stay the same. Passive approach.
		if (loss <= 0.){
			continue;
		}
			
		double probability = std::min(a,loss)/b;
		std::bernoulli_distribution distribution(probability);
		if (distribution(generator)){
			// Calculate the Lagrange Multiplier.
			double Lagrange_Muliplier;
			if(regularization=="none"){
				Lagrange_Muliplier = loss / kernel->HilbertDot(point,point) ;
			}else if (regularization=="l1"){
				Lagrange_Muliplier = std::min(C/probability, loss/kernel->HilbertDot(point,point));
			}else if (regularization=="l2"){
				Lagrange_Muliplier = loss / ( kernel->HilbertDot(point,point) + probability/(2*C) );
			}else{
				//throw exception
				cout << "throw exception" << endl;
			}
			
			// Update the model.
			SVs.push_back(batch.col(i));
			coef.push_back(Lagrange_Muliplier*labels(i));
			//mean_coef.push_back(0.0);
			
			if(SVs.size()>maxSVs){
				SVs.erase(SVs.begin());
				coef.erase(coef.begin());
			}
		
		}
		
	}
	
	cout<<SVs.size()<<endl;
}

arma::dvec KernelPassiveAgressiveClassifier::predict(arma::mat& batch){
	arma::dvec prediction = arma::zeros<arma::dvec>(batch.n_cols);
	for(int i=0;i<batch.n_cols;i++){
		double pred = 0.;
		arma::dvec point = batch.unsafe_col(i);
		for(int j=0;j<SVs.size();j++){
			pred += coef.at(j)*kernel->HilbertDot(SVs.at(j),point);
		}
		if(pred >= 0.){
			prediction(i) = 1.;
		}else{
			prediction(i) = -1.;
		}
	}
	return prediction;
}

inline double KernelPassiveAgressiveClassifier::predict(arma::dvec& data_point){
	double pred = 0.;
	for(int i=0;i<SVs.size();i++){
		pred += coef.at(i)*kernel->HilbertDot(SVs.at(i),data_point);
	}
	if(pred >= 0.){
		pred = 1.;
	}else{
		pred = -1.;
	}
	return pred;
}

inline double KernelPassiveAgressiveClassifier::Margin(arma::dvec& data_point){
	double pred = 0.;
	for(int i=0;i<SVs.size();i++){
		pred += coef.at(i)*kernel->HilbertDot(SVs.at(i),data_point);
	}
	return pred;
}

inline vector<double> KernelPassiveAgressiveClassifier::accuracy(arma::mat& testbatch, arma::dvec& labels){
	
	// Calculate accuracy.
	int errors=0;
	for(int i=0;i<labels.n_elem;i++){
		signed int prediction;
		double pred = 0.;
		arma::dvec point = testbatch.unsafe_col(i);
		for(int j=0;j<SVs.size();j++){
			pred += coef.at(j)*kernel->HilbertDot(SVs.at(j),point);
			//pred += mean_coef.at(j)*coef.at(j)*kernel->HilbertDot(SVs.at(j),point);
		}
		//pred /= 600000;
		if(pred>=0.){
			prediction = 1.;
		}else{
			prediction = -1.;
		}
		if (labels(i)!=prediction){
			errors++;
		}
	}
	
	vector<double> score;
	score.push_back(100.0*(labels.n_elem-errors)/(labels.n_elem));
	score.push_back((double)errors);
	return score;
	
}


/*********************************************
	Multi Layer Perceptron Classifier
*********************************************/

class MLP_Classifier{
protected:
	string name = "Multilayer Perceptron Classifier.";
	Json::Value root; // JSON file to read the hyperparameters.  [optional]
	FFN<NegativeLogLikelihood<>>* model;
	
	double stepSize;
	int batchSize;
	double beta1;
	double beta2;
	double eps;
	int maxIterations;
	double tolerance;
	//Adam* optimizer;
	SGD<MomentumUpdate>* optimizer;
	
public:
	MLP_Classifier(const string cfg);
	
	inline void fit(arma::mat& batch, arma::mat& labels);
	arma::mat predictOnBatch(arma::mat& batch);
	inline double predict(arma::mat& data_point);
	inline vector<double> accuracy(arma::mat& testbatch, arma::mat& labels);
	inline arma::mat& getModel();
	inline SGD<MomentumUpdate>* Opt();
	inline int batch_size();
	inline void InitializePolicy();
	
};

MLP_Classifier::MLP_Classifier(const string cfg){
	try{
		
		std::ifstream cfgfile(cfg);
		cfgfile >> root;
		
		int sz_input_layer = root["NN_Classifier"]
							 .get("Size_of_input_layer",-1).asInt(); 
		if(sz_input_layer <= 0){
			cout << endl <<"Invalid size for input layers." << endl;
			cout << "Parameter Size_of_input_layer must be a positive integer." << endl;
			throw;
		}
		
		int num_hidden_lrs = root["NN_Classifier"]
							 .get("Number_of_hidden_layers",-1).asInt();
		if(num_hidden_lrs <= 0){
			cout << endl <<"Invalid parameter number of hidden layers." << endl;
			cout << "Parameter Number_of_hidden_layers must be a positive integer." << endl;
			throw;
		}
		
		stepSize = root["NN_Classifier"]
				   .get("stepSize",-1).asDouble();
		if(stepSize <= 0.){
			cout << endl <<"Invalid parameter stepSize given." << endl;
			cout << "Parameter stepSize must be a positive double." << endl;
			throw;
		}
		
		batchSize = root["NN_Classifier"]
					.get("batchSize",-1).asInt();
		if(batchSize <= 0 && batchSize >=5000){
			cout << endl <<"Invalid parameter batchSize given." << endl;
			cout << "Parameter batchSize must be a positive integer." << endl;
			throw;
		}
		
		beta1 = root["NN_Classifier"]
				.get("beta1",-1).asDouble();
		if(beta1 <= 0.){
			cout << endl <<"Invalid parameter beta1 given." << endl;
			cout << "Parameter beta1 must be a positive double." << endl;
			throw;
		}
		
		beta2 = root["NN_Classifier"]
				.get("beta2",-1).asDouble();
		if(beta2 <= 0.){
			cout << endl <<"Invalid parameter beta2 given." << endl;
			cout << "Parameter beta2 must be a positive double." << endl;
			throw;
		}
		
		eps = root["NN_Classifier"]
			  .get("eps",-1).asDouble();
		if(eps <= 0.){
			cout << endl <<"Invalid parameter eps given." << endl;
			cout << "Parameter eps must be a positive double." << endl;
			throw;
		}
		
		maxIterations = root["NN_Classifier"]
						.get("maxIterations",-1).asInt();
		if(maxIterations <= 0 ){
			cout << endl <<"Invalid parameter maxIterations given." << endl;
			cout << "Parameter maxIterations must be a positive integer." << endl;
			throw;
		}
		
		tolerance = root["NN_Classifier"]
					.get("tolerance",-1).asDouble();
		if(tolerance <= 0.){
			cout << endl <<"Invalid parameter tolerance given." << endl;
			cout << "Parameter tolerance must be a positive double." << endl;
			throw;
		}
		
		vector<int> layer_size;
		vector<string> layer_activation;
		for(unsigned i = 0; i < num_hidden_lrs; ++i){
			
			layer_size.push_back(root["NN_Classifier"]
								 .get("hidden"+std::to_string(i+1),-1).asInt());
			if(layer_size.back()<=0){
				cout << endl << "Invalid size of hidden layer " << i+1 << "." << endl;
				cout << "Size of hidden layer must be a positive integer." << endl;
				throw;
			}
			
			layer_activation.push_back(root["NN_Classifier"]
									   .get("hidden"+std::to_string(i+1)+"_Activation",-1).asString());
			if(layer_activation.back() != "logistic" &&
			   layer_activation.back() != "relu" &&
			   layer_activation.back() != "softplus" &&
			   layer_activation.back() != "softsign" &&
			   layer_activation.back() != "swish" &&
			   layer_activation.back() != "tanh"){
				cout << endl << "Invalid activation function for layer " << i+1 << "." << endl;
				cout << 
					 "Valid activation functions are : ['identity', 'logistic', 'relu', 'softplus', 'softsign', 'swish', 'tanh']." 
					 << endl;
				throw;
			}
		}
		
		// Built the model
		model = new FFN<NegativeLogLikelihood<>>();
		for(unsigned i=0;i<num_hidden_lrs;i++){
			
			if(i==0){
				model->Add<Linear<> >(sz_input_layer,layer_size.at(i));
			}else{
				model->Add<Linear<> >(layer_size.at(i-1),layer_size.at(i));
			}
			
			if(layer_activation.at(i) == "logistic"){
				model->Add<SigmoidLayer<> >();
			}else if(layer_activation.at(i) == "tanh"){
				model->Add<TanHLayer<> >();
			}else if(layer_activation.at(i) == "relu"){
				model->Add<ReLULayer<> >();
			}else{
				cout << endl << "No support for this activation function at the moment." << endl;
				throw;
			}
			
		}
		model->Add<Linear<> >(layer_size.at(num_hidden_lrs-1), 2);
		model->Add<LogSoftMax<> >();
		model->ResetParameters();
		
		//optimizer = new Adam(stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance, false);
		optimizer = new SGD<MomentumUpdate>(stepSize, batchSize, maxIterations, tolerance, false);
		optimizer->ResetPolicy() = false;
		
	}catch(...){
		throw;
	}

}

inline void MLP_Classifier::fit(arma::mat& batch, arma::mat& labels){
	model->Train( batch, labels, *optimizer );
}

arma::mat MLP_Classifier::predictOnBatch(arma::mat& batch){
	
	arma::mat predictionTemp;
	model->Predict(batch, predictionTemp);
	
	arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);
	for(size_t i = 0; i < predictionTemp.n_cols; ++i){
		prediction(0,i) = arma::as_scalar( arma::find( arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1) );
	}
	
	return prediction;
	
}

inline double MLP_Classifier::predict(arma::mat& data_point){
	
	// Check for invalid data point given
	if( data_point.n_cols < 0 || data_point.n_cols > 1){
		return -1;
	}
	
	arma::mat predictionTemp;
	model->Predict(data_point, predictionTemp);
	
	double prediction = arma::as_scalar( arma::find( arma::max(predictionTemp.col(0)) == predictionTemp.col(0), 1) );
	return prediction;
	
}

inline vector<double> MLP_Classifier::accuracy(arma::mat& testbatch, arma::mat& labels){
	
	arma::mat predictionTemp;
	model->Predict(testbatch, predictionTemp);
	
	arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);
	for(size_t i = 0; i < predictionTemp.n_cols; ++i){
		prediction(0,i) = arma::as_scalar( arma::find( arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1) )  +1;
	}
	
	int errors = 0;
	for(unsigned i = 0; i < prediction.n_cols; ++i){
		if(labels(0,i) != prediction(0,i)){
			errors++;
		}
	}
	
	vector<double> score;
	score.push_back(100.0*(labels.n_elem-errors)/(labels.n_elem));
	score.push_back((double)errors);
	
	return score;
}

inline arma::mat& MLP_Classifier::getModel(){ return model->Parameters(); }

inline SGD<MomentumUpdate>* MLP_Classifier::Opt() { return optimizer; }

inline int MLP_Classifier::batch_size() { return batchSize; }

inline void MLP_Classifier::InitializePolicy() { optimizer->UpdatePolicy().Initialize(model->Parameters().n_rows, model->Parameters().n_cols); }


/*********************************************
	Multi Layer Perceptron Classifier
*********************************************/

class CNN_Classifier{
protected:
	string name = "Convolutional Neural Network Classifier.";
	Json::Value root; // JSON file to read the hyperparameters.  [optional]
	FFN<NegativeLogLikelihood<>, GaussianInitialization>* model;
	
	double stepSize;
	int batchSize;
	double beta1;
	double beta2;
	double eps;
	int maxIterations;
	double tolerance;
	//Adam* optimizer;
	SGD<AdamUpdate>* opt; // The Adam optimizer.
	
public:
	CNN_Classifier(const string cfg);
	
	inline void fit(arma::mat& batch, arma::mat& labels);
	arma::mat predictOnBatch(arma::mat& batch);
	inline double predict(arma::mat& data_point);
	inline vector<double> accuracy(arma::mat& testbatch, arma::mat& labels);
	inline arma::mat& getModel();
	inline SGD<AdamUpdate>* Opt();
	inline int batch_size();
	inline void InitializePolicy();
	
};

CNN_Classifier::CNN_Classifier(const string cfg){
	try{
		
		std::ifstream cfgfile(cfg);
		cfgfile >> root;
		
		stepSize = root["CNN_Classifier"]
				   .get("stepSize",-1).asDouble();
		if(stepSize <= 0.){
			cout << endl <<"Invalid parameter stepSize given." << endl;
			cout << "Parameter stepSize must be a positive double." << endl;
			throw;
		}
		
		batchSize = root["CNN_Classifier"]
					.get("batchSize",-1).asInt();
		if(batchSize <= 0 && batchSize >=5000){
			cout << endl <<"Invalid parameter batchSize given." << endl;
			cout << "Parameter batchSize must be a positive integer." << endl;
			throw;
		}
		
		beta1 = root["CNN_Classifier"]
				.get("beta1",-1).asDouble();
		if(beta1 <= 0.){
			cout << endl <<"Invalid parameter beta1 given." << endl;
			cout << "Parameter beta1 must be a positive double." << endl;
			throw;
		}
		
		beta2 = root["CNN_Classifier"]
				.get("beta2",-1).asDouble();
		if(beta2 <= 0.){
			cout << endl <<"Invalid parameter beta2 given." << endl;
			cout << "Parameter beta2 must be a positive double." << endl;
			throw;
		}
		
		eps = root["CNN_Classifier"]
			  .get("eps",-1).asDouble();
		if(eps <= 0.){
			cout << endl <<"Invalid parameter eps given." << endl;
			cout << "Parameter eps must be a positive double." << endl;
			throw;
		}
		
		maxIterations = root["CNN_Classifier"]
						.get("maxIterations",-1).asInt();
		if(maxIterations <= 0 ){
			cout << endl <<"Invalid parameter maxIterations given." << endl;
			cout << "Parameter maxIterations must be a positive integer." << endl;
			throw;
		}
		
		tolerance = root["CNN_Classifier"]
					.get("tolerance",-1).asDouble();
		if(tolerance <= 0.){
			cout << endl <<"Invalid parameter tolerance given." << endl;
			cout << "Parameter tolerance must be a positive double." << endl;
			throw;
		}
		
		// Built the model
		model = new FFN<NegativeLogLikelihood<>, GaussianInitialization>();
		model->Add<Convolution<> >(1, 32, 5, 5, 1, 1, 2, 2, 28, 28);
		model->Add<ReLULayer<> >();
		model->Add<MaxPooling<> >(2,2,2,2);
		model->Add<Convolution<> >(32, 64, 5, 5, 1, 1, 2, 2, 14, 14);
		model->Add<ReLULayer<> >();
		model->Add<MaxPooling<> >(2,2,2,2);
		model->Add<Linear<> >(7*7*64,1024);
		model->Add<ReLULayer<> >();
		model->Add<Dropout<> >();
		model->Add<Linear<> >(1024,10);
		//model->Add<LogSoftMax<> >();
	
		model->ResetParameters();
		cout << arma::size(model->Parameters()) << endl;
		opt = new SGD<AdamUpdate>(stepSize, batchSize, maxIterations, tolerance, false);
		opt->UpdatePolicy() = AdamUpdate(eps, beta1, beta2);
		opt->ResetPolicy() = false;
		InitializePolicy();
	}catch(...){
		throw;
	}
}

inline void CNN_Classifier::fit(arma::mat& batch, arma::mat& labels){
	model->Train( batch, labels, *opt );
}

arma::mat CNN_Classifier::predictOnBatch(arma::mat& batch){
	
	arma::mat predictionTemp;
	model->Predict(batch, predictionTemp);
	
	arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);
	for(size_t i = 0; i < predictionTemp.n_cols; ++i){
		prediction(0,i) = arma::as_scalar( arma::find( arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1) );
	}
	
	return prediction;
	
}

inline double CNN_Classifier::predict(arma::mat& data_point){
	
	// Check for invalid data point given
	if( data_point.n_cols < 0 || data_point.n_cols > 1){
		return -1;
	}
	
	arma::mat predictionTemp;
	model->Predict(data_point, predictionTemp);
	
	double prediction = arma::as_scalar( arma::find( arma::max(predictionTemp.col(0)) == predictionTemp.col(0), 1) );
	return prediction;
	
}

inline vector<double> CNN_Classifier::accuracy(arma::mat& testbatch, arma::mat& labels){
	
	arma::mat predictionTemp;
	model->Predict(testbatch, predictionTemp);
	
	arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);
	for(size_t i = 0; i < predictionTemp.n_cols; ++i){
		prediction(0,i) = arma::as_scalar( arma::find( arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1) )  +1;
	}
	
	int errors = 0;
	for(unsigned i = 0; i < prediction.n_cols; ++i){
		if(labels(0,i) != prediction(0,i)){
			errors++;
		}
	}
	
	vector<double> score;
	score.push_back(100.0*(labels.n_elem-errors)/(labels.n_elem));
	score.push_back((double)errors);
	
	return score;
}

inline arma::mat& CNN_Classifier::getModel(){ return model->Parameters(); }

inline SGD<AdamUpdate>* CNN_Classifier::Opt() { return opt; }

inline int CNN_Classifier::batch_size() { return batchSize; }

inline void CNN_Classifier::InitializePolicy() { opt->UpdatePolicy().Initialize(model->Parameters().n_rows, model->Parameters().n_cols); }


/*********************************************
	            Dlib LeNet
*********************************************/

class LeNet{
	
using trnet = loss_multiclass_log<relu<fc<10,
								  dropout<relu<fc<1024,
							      max_pool<2,2,2,2,
							      relu<add_layer<con_<64,5,5,1,1,2,2>,
							      max_pool<2,2,2,2,
							      relu<add_layer<con_<32,5,5,1,1,2,2>,
							      input<matrix<unsigned char>>
							      >>>>>>>>>>>>;
public:
	void initializeTrainer();
	void train(std::vector<matrix<unsigned char>> im,std::vector<unsigned long> lb);
	void printNet();
	double getAccuracy(std::vector<matrix<unsigned char>> im,std::vector<unsigned long> lb, bool prnt);
	void Synch();
	void setDropProb(float prob);
	auto& getNet();
protected:
	trnet net;
	dnn_trainer<trnet,adam>* trainer;
};

void LeNet::initializeTrainer(){
	trainer = new dnn_trainer<trnet,adam>(net,adam(0.,0.9,0.999));
	trainer->be_verbose();
	trainer->set_learning_rate(1e-4);
    trainer->set_min_learning_rate(1e-4);
	trainer->set_learning_rate_shrink_factor(1.);
}

void LeNet::train(std::vector<matrix<unsigned char>> im,std::vector<unsigned long> lb){
	trainer->train_one_step(im, lb);
}

void LeNet::printNet(){
	cout << net << endl;
}

void LeNet::setDropProb(float prob){
	layer<3>(net).layer_details()=dropout_(prob);
}

void LeNet::Synch(){
	trainer->get_net();
}

double LeNet::getAccuracy(std::vector<matrix<unsigned char>> im,std::vector<unsigned long> lb, bool prnt=true){
	net.clean();
	std::vector<unsigned long> prediction = net(im);
	size_t correct=0;
	size_t wrong=0;
	double accuracy;
	for(size_t i=0;i<im.size();i++){
		if(prediction[i]==lb[i])
			++correct;
		else
			++wrong;
	}
	accuracy=correct/(double)(correct+wrong);
	net.clean();
	if(prnt){
		cout << "testing num_right: " << correct << endl;
		cout << "testing num_wrong: " << wrong << endl;
		cout << "testing accuracy:  " << accuracy << endl;
	}
	
	return accuracy;
}

auto& LeNet::getNet(){
	return net;
}

/*********************************************
	        Classifier Wrappers
*********************************************/

class Classification{
public:
	virtual void Train() = 0;
};


/*********************************************
	Simple Classification (PA, Kernel PA)
*********************************************/

template<typename clsfr, typename DtTp>
class Simple_Classification : public Classification{
	typedef boost::shared_ptr<hdf5Source<DtTp>> PointToSource;
protected:

	Json::Value root; // JSON file to read the hyperparameters.
	
	clsfr* Classifier; // The classifier.
	int test_size; // Size of test dataset.
	int start_test; // Starting test data point;
	int epochs; // Number of epochs.
	bool negative_labels; // Set the y=0 labels to -1.
	arma::mat testSet; // Test dataset for validation of the classification algorithm.
	arma::dvec testResponses; // Arma row vector containing the labels of the evaluation set.
	double score; // Save the score of the classifier. (To be used later for hyperparameter searching)
	
	PointToSource DataSource; // Data Source to read the dataset in streaming form.
	
public:

	/* Constructor */
	Simple_Classification(string cfg);
	
	/* Destructor */
	~Simple_Classification();
	
	void Train() override;
	
	// Make a prediction.
	inline arma::dvec MakePrediction(arma::mat& batch) { return Classifier->predict(batch); }
	
	// Make a prediction.
	inline double MakePrediction(arma::dvec& data_point) { return Classifier->predict(data_point); }
	
	// Evaluate the classifier on a test set.
	void getScore(arma::mat& testbatch, arma::dvec& labels);
	
	// Getters.
	inline arma::mat& getTestSet() { return testSet; }
	inline arma::dvec& getTestSetLabels() { return testResponses; }
	inline arma::mat* getPTestSet() { return &testSet; }
	inline arma::dvec* getPTestSetLabels() { return &testResponses; }
	
};

template<typename clsfr, typename DtTp>
Simple_Classification< clsfr, DtTp >::Simple_Classification(string cfg)
{
	try{
		// Parse from JSON file.
		std::ifstream cfgfile(cfg);
		cfgfile >> root;
		
		double test_sz = root["parameters"].get("test_size",-1).asDouble();
		if(test_sz<0.0 || test_sz>1.0){
			cout << endl << "Incorrect parameter test_size." << endl;
			cout << "Acceptable test_size parameters : [double] in range:[0.0, 1.0]" << endl;
			throw;
		}
		
		epochs = root["parameters"].get("epochs",-1).asInt64();
		if(epochs<0){
			cout << endl << "Incorrect parameter epochs." << endl;
			cout << "Acceptable epochs parameters : [positive int]" << endl;
			cout << "Epochs must be in range [1:1e+6]." << endl;
			throw;
		}
		
		negative_labels = root["parameters"].get("negative_labels",-1).asBool();
		if(negative_labels!=true && negative_labels!=false){
			cout << endl << "Incorrect negative labels." << endl;
			cout << "Acceptable negative labels parameters : ['default', 'negative_labels', 'augmented_labels', 'negative_augmented_labels']" << endl;
			throw;
		}
		
		// Initialize the Classifier object.
		Classifier = new clsfr(cfg);

		// Initialize data source.
		DataSource = getPSource<DtTp>(root["data"].get("file_name","No file name given").asString(),
									  root["data"].get("dataset_name","No file dataset name given").asString(),
									  false);
									
		test_size = std::floor(test_sz*DataSource->getDatasetLength());
		std::srand (time(NULL));
		start_test = std::rand()%(DataSource->getDatasetLength()-test_size+1);
		
		testSet = arma::zeros<arma::mat>(DataSource->getDataSize(), test_size); // Test dataset for validation of the classification algorithm.
		testResponses = arma::zeros<arma::dvec>(test_size); // Arma row vector containing the labels of the evaluation set.
		
	}catch(...){
		cout << endl << "Something went wrong in LinearClassification object construction." << endl;
		throw;
	}
}

template<typename clsfr, typename DtTp>
Simple_Classification< clsfr, DtTp >::~Simple_Classification(){ 
	DataSource = nullptr;
}

template<typename clsfr, typename DtTp>
void Simple_Classification< clsfr, DtTp >::Train(){
	vector<DtTp>& buffer = DataSource->getbuffer(); // Initialize the dataset buffer.
	int count = 0; // Count the number of processed elements.
	
	for(int ep = 0; ep < epochs; ep++){
		int tests_points_read = 0;
		int start = 0;
		while(DataSource->isValid()){
		
			arma::mat batch; // Arma column wise matrix containing the data points.
			arma::dvec labels; // Arma row vector containing the labels of the training set.
			
			// Load the batch from the buffer while removing the ids.
			batch = arma::mat(&buffer[0],
							  DataSource->getDataSize(),
							  DataSource->getBufferSize(),
							  false,
							  false);
								   
			// Update the number of processed elements.
			count += DataSource->getBufferSize();
			
			if(count>=start_test && tests_points_read!=test_size){
				
				
				if(tests_points_read==0){
					start = start_test - (count-DataSource->getBufferSize()+1);
				}
				
				if(count < start_test + test_size -1){
					
					testSet.cols(tests_points_read, tests_points_read+DataSource->getBufferSize()-1-start) = 
					batch.cols(start, batch.n_cols-1); // Create the test dataset.
					testResponses.rows(tests_points_read, tests_points_read+DataSource->getBufferSize()-1-start) = 
					arma::conv_to<arma::dvec>::from(trans(testSet.cols(tests_points_read, tests_points_read+DataSource->getBufferSize()-1-start)
																 .row(batch.n_rows-1))); // Get the labels.
					
					batch.shed_cols(start, batch.n_cols-1); // Remove the test dataset from the batch.
					
					tests_points_read += DataSource->getBufferSize() - start;
					start = 0;
					
				}else{
					
					testSet.cols(tests_points_read, test_size-1) = 
					batch.cols(start, start+(test_size-tests_points_read)-1); // Create the test dataset.
					testResponses.rows(tests_points_read, test_size-1) = 
					arma::conv_to<arma::dvec>::from(trans(testSet.cols(tests_points_read, test_size-1)
																 .row(batch.n_rows-1))); // Get the labels.
					
					batch.shed_cols(start, start+(test_size-tests_points_read)-1); // Remove the test dataset from the batch.
					
					tests_points_read = test_size;
					
				}
				
				if(tests_points_read == test_size){
					testSet.shed_row(testSet.n_rows-1);
					if(negative_labels){
						// Replace every -1 label with 1.
						for(size_t i = 0; i < testResponses.n_elem; i++){
							if(testResponses(i) == 0.){
								testResponses(i) = -1.;
							}
						}
					}
				}
				
			}
			
			if(batch.n_cols!=0){
				
				labels = arma::conv_to<arma::dvec>::from(trans(batch.row(batch.n_rows-1))); // Get the labels.
				batch.shed_row(batch.n_rows-1); // Remove the labels.
				
				
				if(negative_labels){
					// Replace every -1 label with 1 if necessary.
					for(int i = 0; i < labels.n_elem; i++){
						if(labels(i) == 0.){
							labels(i) = -1.;
						}
					}
				}
				
				// Starting the online learning on the butch
				Classifier->fit(batch,labels);
				
			}
			
			// Get the next 5000 data points from disk to stream them.
			DataSource->advance();
			cout << "count : " << count << endl;
		}
		
		count = 0;
		DataSource->rewind();
	}
	
}

template<typename clsfr, typename DtTp>
void Simple_Classification< clsfr, DtTp >::getScore(arma::mat& testbatch, arma::dvec& labels){
	vector<double> accuracy = Classifier->accuracy(testbatch,labels);
	
	cout << endl << "tests : " << labels.n_elem << endl;
	cout << "start test : " << start_test << endl;
	cout << "errors : " << (int)accuracy.at(1) << endl;
	cout << "accuracy : " << std::setprecision(6) << accuracy.at(0) << "%" << endl;
}


/*********************************************
	Neural Network Classification
*********************************************/

template<typename clsfr, typename DtTp>
class Neural_Classification : public Classification{
	typedef boost::shared_ptr<hdf5Source<DtTp>> PointToSource;
protected:

	Json::Value root; // JSON file to read the hyperparameters.  [optional]
	
	clsfr* Classifier; // The classifier.
	int epochs; // Number of epochs.
	arma::mat testSet; // Test dataset for validation of the classification algorithm.
	arma::mat testResponses; // Arma row vector containing the labels of the evaluation set.
	double score; // Save the score of the classifier. (To be used later for hyperparameter searching)
	
	PointToSource TrainSource; // Data Source to read the dataset in streaming form.
	PointToSource TestSource; // Data Source to read the dataset in streaming form.
	
public:

	/* Constructor */
	Neural_Classification(string cfg);
	
	/* Destructor */
	~Neural_Classification();
	
	// A method for reading the test dataset from a hdf5 file.
	void CreateTestSet();
	
	// Begin the training process.
	void Train() override;
	
	// Make a prediction.
	inline arma::mat MakePrediction(arma::mat& batch) { return Classifier->predict(batch); }
	
	// Make a prediction.
	inline double MakePrediction(arma::dvec& data_point) { return Classifier->predict(data_point); }
	
	// Evaluate the classifier on a test set.
	void getScore(arma::mat& testbatch, arma::mat& testlabels);
	
	// Getters.
	inline arma::mat& getTestSet() { return testSet; }
	inline arma::mat& getTestSetLabels() { return testResponses; }
	inline arma::mat* getPTestSet() { return &testSet; }
	inline arma::mat* getPTestSetLabels() { return &testResponses; }
	
};

template<typename clsfr, typename DtTp>
Neural_Classification< clsfr, DtTp >::Neural_Classification(string cfg)
{
	try{
		
		// Parse from JSON file.
		std::ifstream cfgfile(cfg);
		cfgfile >> root;
		std::srand (time(NULL));
		
		epochs = root["parameters"].get("epochs",-1).asInt64();
		if(epochs<0){
			cout << endl << "Incorrect parameter epochs." << endl;
			cout << "Acceptable epochs parameters : [positive int]" << endl;
			cout << "Epochs must be in range [1:1e+6]." << endl;
			throw;
		}
		
		size_t testSize = root["parameters"].get("test_size",-1).asInt64();
		if(epochs<0){
			cout << endl << "Incorrect parameter epochs." << endl;
			cout << "Acceptable epochs parameters : [positive int]" << endl;
			cout << "Epochs must be in range [1:1e+6]." << endl;
			throw;
		}
		
		// Initialize the Classifier object.
		Classifier = new clsfr(cfg);

		// Initialize data source.
		TrainSource = getPSource<DtTp>(root["data"].get("file_name","No file name given").asString(),
									   root["data"].get("train_dset_name","No file dataset name given").asString(),
									   false);
									   
		// Initialize data source.
		TestSource = getPSource<DtTp>(root["data"].get("file_name","No file name given").asString(),
								      root["data"].get("test_dset_name","No file dataset name given").asString(),
								      false);
		
		testSet = arma::zeros<arma::mat>(TrainSource->getDataSize()-1, testSize); // Test dataset for validation of the classification algorithm.
		testResponses = arma::zeros<arma::mat>(1, testSize); // Arma row vector containing the labels of the evaluation set.
		CreateTestSet();
		
	}catch(...){
		cout << endl << "Something went wrong in Neural Classification object construction." << endl;
		throw;
	}
}

template<typename clsfr, typename DtTp>
Neural_Classification< clsfr, DtTp >::~Neural_Classification(){ 
	TrainSource = nullptr;
	TestSource = nullptr;
}

template<typename clsfr, typename DtTp>
void Neural_Classification< clsfr, DtTp >::CreateTestSet(){
	
	std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long>         training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long>         testing_labels;
    load_mnist_dataset("/home/aris/Desktop/IEEE_Machine_Learning/MNIST_Data", training_images, training_labels, testing_images, testing_labels);
	
	std::vector<matrix<unsigned char>>::iterator it;
	for(it=training_images.begin();it!=training_images.end();++it){
		cout << it[0] << endl << endl;
	}
	
	vector<DtTp>& buffer = TestSource->getbuffer(); // Initialize the dataset buffer.
	int position = TestSource->getCurrentPos()-TestSource->getBufferSize();
	while(TestSource->isValid()){
		arma::mat batch = arma::mat(&buffer[0],
						            TestSource->getDataSize(),
									TestSource->getBufferSize(),
									false,
									false);
		
		testResponses.cols(position, position+TestSource->getBufferSize()-1) = batch.row(batch.n_rows-1);
	    batch.shed_row(batch.n_rows-1);
		testSet.cols(position, position+TestSource->getBufferSize()-1) = batch.cols(0,batch.n_cols-1);
		position+=TestSource->getBufferSize();
		
		// Get the next 1000 data points from disk to stream them.
		TestSource->advance();
	}
	
}

template<typename clsfr, typename DtTp>
void Neural_Classification< clsfr, DtTp >::Train(){
	vector<DtTp>& buffer = TrainSource->getbuffer(); // Initialize the dataset buffer.
	int count = 0; // Count the number of processed elements.
	int trained_on = 0;
	
	for(int ep = 0; ep < epochs; ep++){
		while(TrainSource->isValid()){
			arma::mat batch; // Arma column wise matrix containing the data points.
			arma::mat labels; // Arma row vector containing the labels of the training set.
			
			// Load the batch from the buffer while removing the ids.
			batch = arma::mat(&buffer[0],
							  TrainSource->getDataSize(),
							  TrainSource->getBufferSize(),
							  false,
							  false);
								   
			// Update the number of processed elements.
			count += TrainSource->getBufferSize();
				
			// batch = arma::shuffle(batch, 1);
			labels = arma::conv_to<arma::mat>::from(batch.row(batch.n_rows-1)); // Get the labels.
			batch.shed_row(batch.n_rows-1); // Remove the labels.
			
			// Do batch learning. The batch can be an integer in the interval [1:5000].
			size_t mod = batch.n_cols % Classifier->batch_size();
			size_t num_of_batches = std::floor( batch.n_cols / Classifier->batch_size() );
			
			if( num_of_batches > 0 ){
				for(unsigned iter = 0; iter < num_of_batches; iter++){
					arma::mat stream_points = arma::mat(&batch.unsafe_col(iter*Classifier->batch_size())(0),
																		  batch.n_rows, Classifier->batch_size(),
																		  false);
					arma::mat stream_labels = arma::mat(&labels.unsafe_col(iter*Classifier->batch_size())(0),
																		   1,
																		   Classifier->batch_size(),
																		   false);
					Classifier->fit(stream_points, stream_labels);
				}
				if( mod > 0 ){
					arma::mat stream_points = arma::mat(&batch.unsafe_col(batch.n_cols - mod)(0) , batch.n_rows, mod, false);
					arma::mat stream_labels = arma::mat(&labels.unsafe_col(batch.n_cols - mod)(0), 1, mod, false);
					Classifier->fit(stream_points, stream_labels);
				}
			}else{
				Classifier->fit(batch,labels);
			}
			
			trained_on += batch.n_cols;
			
			// Get the next 1000 data points from disk to stream them.
			TrainSource->advance();
			cout << "count : " << count << endl;
			if(count%100000==0){
				getScore(testSet, testResponses);
			}
		}
		
		cout << endl << "Epoch " << ep << " completed." << endl;
		cout << "////////////////////////////////////////////";
		getScore(testSet, testResponses);
		
		count = 0;
		TrainSource->rewind();
	}
	cout << "Trained on " << trained_on << " datapoints." << endl;
}

template<typename clsfr, typename DtTp>
void Neural_Classification< clsfr, DtTp >::getScore(arma::mat& testbatch, arma::mat& testlabels){
	size_t mod = testlabels.n_cols % Classifier->batch_size();
	size_t num_of_batches = std::floor( testlabels.n_cols / Classifier->batch_size() );
	if(num_of_batches>0){
		double errors = 0;
		for(unsigned iter = 0; iter < num_of_batches; iter++){
			arma::mat test_points = arma::mat(&testbatch.unsafe_col(iter*Classifier->batch_size())(0),
															        testbatch.n_rows, Classifier->batch_size(),
															        false);
			arma::mat test_labels = arma::mat(&testlabels.unsafe_col(iter*Classifier->batch_size())(0),
															         1,
																	 Classifier->batch_size(),
															         false);
			errors += (int)Classifier->accuracy(test_points, test_labels).at(1);
		}
		if( mod > 0 ){
			arma::mat test_points = arma::mat(&testbatch.unsafe_col(testbatch.n_cols - mod)(0) , testbatch.n_rows, mod, false);
			arma::mat test_labels = arma::mat(&testlabels.unsafe_col(testlabels.n_cols - mod)(0), 1, mod, false);
			errors += (int)Classifier->accuracy(test_points, test_labels).at(1);
		}
		cout << endl << "tests : " << testlabels.n_elem << endl;
		cout << "errors : " << errors << endl;
		cout << "accuracy : " << std::setprecision(6) << 100.0*(testlabels.n_elem-errors)/(testlabels.n_elem) << "%" << endl;
	}else{
		vector<double> accuracy = Classifier->accuracy(testbatch, testlabels);
		cout << endl << "tests : " << testlabels.n_elem << endl;
		cout << "errors : " << (int)accuracy.at(1) << endl;
		cout << "accuracy : " << std::setprecision(6) << accuracy.at(0) << "%" << endl;
	}
	
}


/*********************************************
	Neural Network Classification
*********************************************/

class LeNet_Classification : public Classification{
	typedef boost::shared_ptr<hdf5Source<double>> PointToSource;
protected:

	Json::Value root; // JSON file to read the hyperparameters.  [optional]
	
	int epochs; // Number of epochs.
	arma::mat testSet; // Test dataset for validation of the classification algorithm.
	arma::mat testResponses; // Arma row vector containing the labels of the evaluation set.
	double score; // Save the score of the classifier. (To be used later for hyperparameter searching)
	
	std::vector<matrix<unsigned char>> training_images;
	std::vector<unsigned long> training_labels;
	std::vector<matrix<unsigned char>> test_images;
	std::vector<unsigned long> test_labels;
	
	PointToSource TrainSource; // Data Source to read the dataset in streaming form.
	PointToSource TestSource; // Data Source to read the dataset in streaming form.
	
	LeNet net;
	resizable_tensor parameters;
	size_t num_of_par; // The total number of trainable parameters.
	
	vector<float> AccuracyHistory;
	vector<int> DatapointsProcessed;
	
public:

	/* Constructor */
	LeNet_Classification(string cfg);
	
	/* Destructor */
	~LeNet_Classification();
	
	// A method for reading the test dataset from a hdf5 file.
	void CreateTestSet();
	
	// A method for collecting the networks parameters.
	void CreateParams();
	
	// Print useful info of the parameters.
	void PrintParams();
	
	// Begin the training process.
	void Train() override;
	
	// Make a prediction.
	//inline arma::mat MakePrediction(arma::mat& batch) { return Classifier->predict(batch); }
	
	// Make a prediction.
	//inline double MakePrediction(arma::dvec& data_point) { return Classifier->predict(data_point); }
	
	// Evaluate the classifier on a test set.
	void getScore(arma::mat& testbatch, arma::mat& testlabels);
	
	// Getters.
	inline arma::mat& getTestSet() { return testSet; }
	inline arma::mat& getTestSetLabels() { return testResponses; }
	inline arma::mat* getPTestSet() { return &testSet; }
	inline arma::mat* getPTestSetLabels() { return &testResponses; }
	
};

LeNet_Classification::LeNet_Classification(string cfg)
{
	try{
		
		// Parse from JSON file.
		std::ifstream cfgfile(cfg);
		cfgfile >> root;
		std::srand (time(NULL));
		
		epochs = root["parameters"].get("epochs",-1).asInt64();
		if(epochs<0){
			cout << endl << "Incorrect parameter epochs." << endl;
			cout << "Acceptable epochs parameters : [positive int]" << endl;
			cout << "Epochs must be in range [1:1e+6]." << endl;
			throw;
		}
		
		size_t testSize = root["parameters"].get("test_size",-1).asInt64();
		if(epochs<0){
			cout << endl << "Incorrect parameter epochs." << endl;
			cout << "Acceptable epochs parameters : [positive int]" << endl;
			cout << "Epochs must be in range [1:1e+6]." << endl;
			throw;
		}

		// Initialize data source.
		TrainSource = getPSource<double>(root["data"].get("file_name","No file name given").asString(),
									   root["data"].get("train_dset_name","No file dataset name given").asString(),
									   false);
									   
		// Initialize data source.
		TestSource = getPSource<double>(root["data"].get("file_name","No file name given").asString(),
								      root["data"].get("test_dset_name","No file dataset name given").asString(),
								      false);
		
		testSet = arma::zeros<arma::mat>(TrainSource->getDataSize()-1, testSize); // Test dataset for validation of the classification algorithm.
		testResponses = arma::zeros<arma::mat>(1, testSize); // Arma row vector containing the labels of the evaluation set.
		CreateTestSet();
		net.initializeTrainer();
		net.printNet();
		
	}catch(...){
		cout << endl << "Something went wrong in Neural Classification object construction." << endl;
		throw;
	}
}

LeNet_Classification::~LeNet_Classification(){ 
	TrainSource = nullptr;
	TestSource = nullptr;
}

void LeNet_Classification::CreateParams(){
	
	tensor* par_fc2 = &layer<2>(net.getNet()).layer_details().get_layer_params();
	tensor* par_fc1 = &layer<5>(net.getNet()).layer_details().get_layer_params();
	tensor* par_conv2 = &layer<8>(net.getNet()).layer_details().get_layer_params();
	tensor* par_conv1 = &layer<11>(net.getNet()).layer_details().get_layer_params();
	
	vector<tensor*> params;
	params.push_back(par_conv1);
	params.push_back(par_conv2);
	params.push_back(par_fc1);
	params.push_back(par_fc2);
	
	num_of_par=0;
	for(size_t i=0;i<params.size();i++){
		num_of_par+=params[i]->size();
	}
	parameters.set_size(num_of_par,1,1,1);
	
	size_t counter = 0;
	for(size_t i=0;i<params.size();i++){
		for(auto it=params[i]->begin();it!=params[i]->end();++it){
			parameters.host()[i+counter]=*it;
		}
		counter+=params[i]->size();
	}
}

void LeNet_Classification::PrintParams(){
	cout<<"Change "<<parameters.host()[0]<<endl;
}

void LeNet_Classification::CreateTestSet(){
	vector<double>& buffer = TestSource->getbuffer(); // Initialize the dataset buffer.
	while(TestSource->isValid()){
		for(size_t i=0;i<buffer.size();){
			matrix<unsigned char,28,28> image;
			unsigned long label;
			for(size_t j=0;j<28;j++){
				for(size_t k=0;k<28;k++){
					image(j,k)=(unsigned char)buffer.at(i);
					++i;
				}
			}
			label=(unsigned long)(buffer.at(i)-1);
			++i;
			test_images.push_back(image);
			test_labels.push_back(label);
		}
		TestSource->advance();
	}
}

void LeNet_Classification::Train(){
	// Initialize the dataset buffer.
	vector<double>& buffer = TrainSource->getbuffer(); 
	std::vector<matrix<unsigned char>> mini_batch_samples;
    std::vector<unsigned long> mini_batch_labels;
	size_t count = 0;
	double accuracy = 0;
	bool init = false;
	cout << endl << "Size of Streaming Dataset : " << TrainSource->getDatasetLength() << endl << endl;
	
	string filename = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/Graphs/CentralizedLeNet.csv";
	std::ofstream myfile;
	
	for(size_t ep=0; ep<1; ep++){
		
		cout << "Starting epoch " << ep+1 << endl;
		size_t learning_round = 0;
		while(TrainSource->isValid()){
			training_images.clear();
			training_labels.clear();
			for(size_t i=0;i<buffer.size();){
				matrix<unsigned char,28,28> image;
				unsigned long label;
				for(size_t j=0;j<28;j++){
					for(size_t k=0;k<28;k++){
						image(j,k)=(unsigned char)buffer.at(i);
						++i;
					}
				}
				label=(unsigned long)(buffer.at(i)-1);
				++i;
				training_images.push_back(image);
				training_labels.push_back(label);
			}
			
			size_t posit=0;
			bool done=false;
			while(posit<training_images.size()&&(!done)){
				
				// Create the batch.
				mini_batch_samples.clear();
				mini_batch_labels.clear();
				while(mini_batch_samples.size()<64){
					if(posit<training_images.size()){
						mini_batch_samples.push_back(training_images[posit]);
						mini_batch_labels.push_back(training_labels[posit]);
						posit++;
					}else{
						done=true;
						break;
					}
				}
				
				// Fit the batch ten times.
				for(size_t fits=0;fits<1;fits++){
					net.train(mini_batch_samples,mini_batch_labels);
				}
				net.Synch();
				count+=mini_batch_samples.size();
				learning_round+=1;
				
				if(!init){
					CreateParams();
					//PrintParams();
					init=true;
				}
				
				// Print info.
				cout<<endl;
				if(learning_round%10==0){
					net.setDropProb(0.);
					accuracy = net.getAccuracy(test_images,test_labels);
					net.setDropProb(0.5);
					//AccuracyHistory.push_back(accuracy);
	                //DatapointsProcessed.push_back(count);
					myfile.open(filename, std::ios::app);
					myfile << accuracy << "," << count << "\n";
					myfile.close();
				}
				else{
					cout << "Accuracy: " << accuracy << endl;	
				}
				cout << "Fitted: " << count << endl;
			}
			TrainSource->advance();
			PrintParams();
 		}
		TrainSource->rewind();
		
		net.setDropProb(0.);
		accuracy=net.getAccuracy(test_images,test_labels,false);
		net.setDropProb(0.5);
		cout << "Epochs completed " << ep+1 << endl;
		cout << "Accuracy: " << accuracy << endl;
		cout << "Dara fitted: " << count << endl;
	}
	
	
	//for(size_t i=0;i<AccuracyHistory.size();i++){
		//myfile << AccuracyHistory.at(i) << "," << DatapointsProcessed.at(i) << "\n";
	//}
	

}

} // end of namespace ML_Classification
#endif
