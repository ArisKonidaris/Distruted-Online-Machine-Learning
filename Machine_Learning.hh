#ifndef _MACHINE_LEARNING_HH_
#define _MACHINE_LEARNING_HH_

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <jsoncpp/json/json.h>
#include <cmath>
#include <time.h>
#include <set>

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/core/optimizers/ada_grad/ada_grad.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/core/optimizers/adam/adam.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
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

namespace machine_learning{
	
using std::string;
using std::cout;
using std::endl;
using std::vector;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using dlib::loss_multiclass_log;
using dlib::relu;
using dlib::fc;
using dlib::dropout;
using dlib::dropout_;
using dlib::max_pool;
using dlib::add_layer;
using dlib::con_;
using dlib::matrix;
using dlib::input;
using dlib::resizable_tensor;
using dlib::tensor;
using dlib::dnn_trainer;
using dlib::adam;
using dlib::layer;
	
class MLPACK_Learner{
protected:
	Json::Value root; // JSON file to read the hyperparameters.
	//arma::mat _model; // The model parameters in matrix form.
	vector<arma::mat*> vector_model; // The models' parameter layers in a vector.
	size_t numberOfUpdates; // The number of time the parameters have been updated.
public:
	MLPACK_Learner() { }
	
	inline size_t getNumOfUpdates(){
		size_t updates = numberOfUpdates;
		numberOfUpdates = 0;
		return updates;
	}
	
	// All these functions must be overridden by its descendants
	virtual void initializeModel(size_t sz) { }
	virtual void initializeModel(size_t sz1, size_t sz2) { }
	//virtual void update_model(arma::mat w) { }
	//virtual void update_model_by_ref(const arma::mat& w) { }
	virtual void update_model(const vector<arma::mat>& w) { }
	virtual void handleVD(size_t sz) { }
	virtual void handleRD(size_t sz) { }
	virtual void fit(const arma::mat& point, const arma::mat& label) { }
	virtual arma::mat predict(const arma::mat& point) const { return nullptr; }
	//virtual inline arma::mat& getModel() { return _model; }
	virtual inline vector<arma::mat*>& getModel() { return vector_model; }
	virtual inline vector<arma::mat*>& getHModel() { vector<arma::mat*> empt_vec; return empt_vec; }
	virtual inline void restoreModel(const vector<arma::mat*>& params) { }
	virtual inline double accuracy(const arma::mat& testbatch, const arma::mat& labels) const { return 0.; }
	virtual inline size_t byte_size() const { return 0; }
	virtual inline vector<arma::SizeMat> modelDimensions() const { vector<arma::SizeMat> empt_vec; return empt_vec; }
};

template<typename feats,typename lbs>
class DLIB_Learner{
	typedef std::vector<matrix<feats>> input_features;
	typedef std::vector<lbs> input_labels;
protected:
	Json::Value root; // JSON file to read the hyperparameters.
	vector<tensor*> parameters; // The model parameters in matrix form.
	size_t numberOfUpdates; // The number of time the parameters have been updated.
	size_t num_of_params; // The total number of model's learning parameters.
public:
	
	DLIB_Learner() { }
	
	inline size_t getNumOfUpdates(){
		size_t updates = numberOfUpdates;
		numberOfUpdates = 0;
		return updates;
	}
	
	// All these functions must be overridden by its descendants
	virtual void initialize_size() =0;
	virtual void initializeTrainer() =0;
	virtual void update_model(const vector<resizable_tensor>& w) =0;
	virtual void update_model(const vector<resizable_tensor*>& w) =0;
	virtual void fit(const input_features& point, const input_labels& label) =0;
	virtual input_labels predict(const input_features& point) =0;
	virtual inline double accuracy(const input_features& testbatch, const input_labels& labels) { return 0.; }
	virtual inline vector<tensor*>& Parameters() { return parameters; }
	virtual inline size_t byte_size() const =0;
	virtual inline size_t modelDimensions() { return num_of_params; }
};
	
namespace MLPACK_Classification{
	

/*********************************************
	Passive Aggressive Classifier
*********************************************/

class PassiveAgressiveClassifier : public MLPACK_Learner {
protected:
	
	arma::dvec W; // Parameter vector.
	double intercept; // The bias.
	string regularization; // Regularization technique.
	double C; // Regularization C term. [optional]
	
public:
	PassiveAgressiveClassifier(string cfg, string net_name);
	
	// Initialize the parameters of the model.
	void initializeModel(size_t sz) override;
	
	// Update the parameters of the model.
	void update_model(const vector<arma::mat>& w) override;
	
	// Stream update.
	void fit(const arma::mat& point, const arma::mat& labels) override;
	
	// Make a prediction.
	arma::mat predict(const arma::mat& point) const  override;
	
	// Get score.
	inline double accuracy(const arma::mat& testbatch, const arma::mat& labels) const override{
		int errors=0;
		for(size_t i=0;i<labels.n_cols;i++){
			double prediction;
			if( arma::dot(W,testbatch.unsafe_col(i)) + intercept >= 0. ){
				prediction = 1.;
			}else{
				prediction = -1.;
			}
			if (labels(i)!=prediction){
				errors++;
			}
		}
		return (double)errors;
	}
	
	inline vector<arma::mat*>& getModel() override;
	inline vector<arma::SizeMat> modelDimensions() const override;
	inline size_t byte_size() const override { return sizeof(float)*(W.n_elem+1); }
};


/*********************************************
	 Extreme Learning Machine Classifier
*********************************************/

class ELM_Classifier : public MLPACK_Learner {
protected:

	arma::mat A; // Hidden Layer Weights.
	arma::mat b; // Hidden Layer Biases.
	arma::mat beta; // Output Learnable Parameters.
	arma::mat K; // Autocorrelation matrix of hidden layer matrix H.
	size_t num_of_neurons; // The number of neurons the hidden layer has.
	vector<arma::mat*> hidden_parameters; // The vector containing the static hidden variables.
	Json::Value root; // JSON file to read the hyperparameters.

public:

	// Constructor
	ELM_Classifier(string cfg, string net_name);
	
	// Method that initializes the learner.
	void initializeModel(size_t num_of_feats, size_t num_of_classes) override;
	
	// Method that updates the parameters of the model.
	void update_model(const vector<arma::mat>& w) override;
	
	// Method that expands each neuron with sz weights. This happens in case of Virtual Concept Drift.
	void handleVD(size_t sz);
	
	// Method that expands the beta vector by sz parameters. This happens in case of Real Concept Drift.
	void handleRD(size_t sz);
	
	// Stream update.
	void fit(const arma::mat& batch, const arma::mat& label) override;
	
	// Make a prediction.
	arma::mat predict(const arma::mat& batch) const override;
	
	// Get score.
	inline double accuracy(const arma::mat& testbatch, const arma::mat& labels) const;
	
	// Restore the model using variables from a pre-trained model.
	inline void restoreModel(const vector<arma::mat*>& params) override;
	
	// Retrieve the hidden variables.
	inline vector<arma::mat*>& getHModel() override;
	
	// Get the arma sizes of the model.
	inline vector<arma::SizeMat> modelDimensions() const override;
	
	// The size of the model. Used to calculate the size of the TCP packet.
	inline size_t byte_size() const override { return sizeof(float)*(K.n_elem+beta.n_elem); }

};


/*********************************************
	Multi Layer Perceptron Classifier
*********************************************/

class MLP_Classifier : public MLPACK_Learner {
protected:
	FFN<NegativeLogLikelihood<>, GaussianInitialization>* model;
	
	double stepSize; // The step size / learning rate of the selected optimizer.
	double beta1; // Adam optimizer parameter b1. If another optmizer is selected this variable remains undefined.
	double beta2; // Adam optimizer parameter b2. If another optmizer is selected this variable remains undefined.
	double eps; // Value used to initialise the mean squared gradient parameter for the selected optimizer.
	size_t maxIterations; /** The maximum number of points that are processed by the selected optimizer
						  (i.e., one iteration equals one point; one iteration does not equal one pass over the dataset). */
	double tolerance; // The tolerance parameter of the selected optimizer.
	size_t batch_size; // The batch size.
	
	vector<int> layer_size; // A vector where each value refers to the number of neurons of the corresponing hidden layer.
	vector<string> layer_activation; // The activation function of the corresponding hidden layer.
	SGD<AdamUpdate>* opt; // The Adam optimizer.
	
public:
	MLP_Classifier(string cfg, string net_name);
	
	// Initialize the parameters of the model.
	void initializeModel(size_t sz) override;
	
	// Update the parameters of the model.
	void update_model(const vector<arma::mat>& w) override;
	
	// Stream update.
	void fit(const arma::mat& batch, const arma::mat& labels) override;
	
	// Make a prediction.
	arma::mat predict(const arma::mat& batch) const override;
	
	// Get score.
	inline double accuracy(const arma::mat& testbatch, const arma::mat& labels) const override;
	
	inline size_t byte_size() const override { return sizeof(float)*(model->Parameters().n_elem); }
	inline vector<arma::SizeMat> modelDimensions() const override;
	
};

} /** End namespace MLPACK_Classification */

namespace MLPACK_Regression{


/*********************************************
	Passive Aggressive Regressor
*********************************************/

class PassiveAgressiveRegression : public MLPACK_Learner {
protected:
	arma::dvec W; // Parameter vector.
	string regularization; // Regularization technique.
	double C; // Regularization C term. [optional]
	double epsilon; // The epsilon-insensitive parameter.
public:
	PassiveAgressiveRegression(string cfg, string net_name);
	
	// Initialize the parameters of the model.
	void initializeModel(size_t sz) override;
	
	// Update the parameters of the model.
	void update_model(const vector<arma::mat>& w) override;
	
	// Stream update.
	void fit(const arma::mat& point, const arma::mat& labels) override;
	
	// Make a prediction.
	arma::mat predict(const arma::mat& point) const  override;
	
	// Get score.
	inline double accuracy(const arma::mat& testbatch, const arma::mat& labels) const override{
		double RMSE = 0;
		for(size_t i=0;i<labels.n_cols;i++){
			RMSE += std::pow( labels(0,i) - arma::dot(W,testbatch.unsafe_col(i)) , 2);
		}
		cout << endl << "(RMSE*T)^2 = " << RMSE << endl;
		RMSE /= labels.n_cols;
		
		return std::sqrt(RMSE);
	}
	
	inline vector<arma::SizeMat> modelDimensions() const override;
	inline size_t byte_size() const override { return sizeof(float)*(W.n_elem); }
	
};


/*********************************************
	Neural Network Regressor
*********************************************/

class NN_Regressor : public MLPACK_Learner {
protected:
	FFN< MeanSquaredError<>, GaussianInitialization>* model; // The actual feed forward neurar network topology.
	
	double stepSize; // The step size / learning rate of the selected optimizer.
	double beta1; // Adam optimizer parameter b1. If another optmizer is selected this variable remains undefined.
	double beta2; // Adam optimizer parameter b2. If another optmizer is selected this variable remains undefined.
	double eps; // Value used to initialise the mean squared gradient parameter for the selected optimizer.
	size_t maxIterations; /** The maximum number of points that are processed by the selected optimizer
						  (i.e., one iteration equals one point; one iteration does not equal one pass over the dataset). */
	double tolerance; // The tolerance parameter of the selected optimizer.
	size_t batch_size; // The batch size.
	
	vector<int> layer_size; // A vector where each value refers to the number of neurons of the corresponing hidden layer.
	vector<string> layer_activation; // The activation function of the corresponding hidden layer.
	SGD<AdamUpdate>* opt; // The Adam optimizer.
	
public:

	NN_Regressor(string cfg, string net_name);
	
	// Initialize the parameters of the model.
	void initializeModel(size_t sz) override;
	
	// Update the parameters of the model.
	void update_model(const vector<arma::mat>& w) override;
	
	// Stream update
	void fit(const arma::mat& batch, const arma::mat& labels) override;
	
	// Make a prediction.
	arma::mat predict(const arma::mat& batch) const override;
	
	// Get score
	inline double accuracy(const arma::mat& test_data, const arma::mat& labels) const override;

	inline vector<arma::SizeMat> modelDimensions() const override;
	inline size_t byte_size() const override { return sizeof(float)*(model->Parameters().n_elem); }
	
};

} /** End namespace MLPACK_Regression */

namespace DLIB_Classification{
	

/*********************************************
	            Dlib LeNet
*********************************************/

template<typename feats, typename lbs>
class LeNet : public DLIB_Learner<feats,lbs> {
	typedef std::vector<matrix<feats>> input_features;
	typedef std::vector<lbs> input_labels;
	
	using trnet = loss_multiclass_log<relu<fc<10,
								  dropout<relu<fc<256,
								  max_pool<2,2,2,2,
								  relu<add_layer<con_<64,5,5,1,1,2,2>,
								  max_pool<2,2,2,2,
								  relu<add_layer<con_<32,5,5,1,1,2,2>,
								  input<matrix<feats>>
								  >>>>>>>>>>>>;
public:
	
	// Default constructor.
	LeNet(string nodeName)
	:synchFileID(nodeName) {
		this->numberOfUpdates=0; 
		this->num_of_params=0; 
		initializeTrainer(); 
		par_fc2 = &layer<2>(net).layer_details().get_layer_params();
		par_fc1 = &layer<5>(net).layer_details().get_layer_params();
		par_conv2 = &layer<8>(net).layer_details().get_layer_params();
		par_conv1 = &layer<11>(net).layer_details().get_layer_params();
		this->parameters.push_back(par_conv1);
		this->parameters.push_back(par_conv2);
		this->parameters.push_back(par_fc1);
		this->parameters.push_back(par_fc2);
	}

	// Check if the parameter visitors are initialized.
	void initialize_size() override;
	
	// Initialize the trainer of the model.
	void initializeTrainer() override;
	
	// Method for changing the drop probability of the dropout layer.
	void setDropProb(float prob);
	
	// Stream update.
	void fit(const input_features& points, const input_labels& label) override;
	
	// Make a prediction.
	input_labels predict(const input_features& point) override;
	
	// Print the architecture of the LeNet network.
	void printNet();
	
	// Method for calculating the accuracy of the network.
	double accuracy(const input_features& testbatch, const input_labels& labels) override;
	
	// Method that synchronizes trainer and network.
	void Synch();
	
	// Set the parameters of the model.
	void update_model(const vector<resizable_tensor>& w) override;
	void update_model(const vector<resizable_tensor*>& w) override;
	
	// Get the netwoks' parameters.
	inline vector<tensor*>& Parameters() override;
	
	// Get the size of the parameters.
	inline size_t byte_size() const override; 
	
	// Get the LeNet network.
	auto& Model();
	
	// Get the number of the learnable parameters.
	inline size_t modelDimensions() override { return this->num_of_params; }
	
protected:
	trnet net; // The LeNet network.
	dnn_trainer<trnet,adam>* trainer; // The adam trainer.
	size_t maxIterations; // The number of times each batch is fitted.
	tensor* par_conv1; // The parameters of the first convolutional layer.
	tensor* par_conv2; // The parameters of the second convolutional layer.
	tensor* par_fc1; // The parameters of the first fully connected hidden layer.
	tensor* par_fc2; // The parameters of the second fully connected hidden layer.
	string synchFileID; // Node name to be used by the optimizer for thread synchronization.
};

template<typename feats, typename lbs>
void LeNet<feats,lbs>::initialize_size(){
	if(this->num_of_params==0){
		for(auto layer:this->parameters){
			this->num_of_params+=layer->size();
		}
		cout << "The CNN has " << this->num_of_params << " learnable parameters." << endl;
	}
}
	
template<typename feats, typename lbs>
void LeNet<feats,lbs>::initializeTrainer(){
	trainer = new dnn_trainer<trnet,adam>(net,adam(0.,0.9,0.999));
	trainer->be_verbose();
	trainer->set_learning_rate(1e-4);
    trainer->set_min_learning_rate(1e-4);
	trainer->set_learning_rate_shrink_factor(1.);
	maxIterations = 1;
}

template<typename feats, typename lbs>
void LeNet<feats,lbs>::setDropProb(float prob){
	layer<3>(net).layer_details()=dropout_(prob);
}

template<typename feats, typename lbs>
void LeNet<feats,lbs>::fit(const input_features& points, const input_labels& labels){
	for(size_t i=0;i<maxIterations;i++){
		trainer->train_one_step(points, labels);
	}
	Synch();
	this->numberOfUpdates+=points.size();
}

template<typename feats, typename lbs>
std::vector<lbs> LeNet<feats,lbs>::predict(const input_features& point){
	net.clean();
	return net(point);
}

template<typename feats, typename lbs>
void LeNet<feats,lbs>::printNet(){
	cout << net << endl;
}

template<typename feats, typename lbs>
double LeNet<feats,lbs>::accuracy(const input_features& im, const input_labels& lb){
	net.clean();
	setDropProb(0.);
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
	setDropProb(0.5);
	net.clean();
	
	return accuracy;
}

template<typename feats, typename lbs>
void LeNet<feats,lbs>::Synch(){
	trainer->get_net();
}

template<typename feats, typename lbs>
void LeNet<feats,lbs>::update_model(const vector<resizable_tensor>& w){
	for(size_t i=0;i<w.size();i++){
		//dlib:memcpy(*this->parameters.at(i), w.at(i));
		//dlib::cpu::affine_transform(*this->parameters.at(i), w.at(i), 1., 0.);
		dlib::cuda::affine_transform(*this->parameters.at(i), w.at(i), 1.);
	}
	initialize_size();
}

template<typename feats, typename lbs>
void LeNet<feats,lbs>::update_model(const vector<resizable_tensor*>& w){
	for(size_t i=0;i<w.size();i++){
        //dlib:memcpy(*this->parameters.at(i), *w.at(i));
		//dlib::cpu::affine_transform(*this->parameters.at(i), *w.at(i), 1., 0.);
		dlib::cuda::affine_transform(*this->parameters.at(i), *w.at(i), 1.);
	}
	initialize_size();
}

template<typename feats, typename lbs>
vector<tensor*>& LeNet<feats,lbs>::Parameters(){
	return this->parameters;
}

template<typename feats, typename lbs>
size_t LeNet<feats,lbs>::byte_size() const{
	return sizeof(float)*this->num_of_params;
}

template<typename feats, typename lbs>
auto& LeNet<feats,lbs>::Model(){
	return net;
}

} /** End namespace DLIB_Classification */

namespace DLIB_Regression{
	
}  /** End namespace DLIB_Regression */

} /** End machine_learning */

#endif
