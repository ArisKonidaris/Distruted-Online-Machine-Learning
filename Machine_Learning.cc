#include "Machine_Learning.hh"

using namespace machine_learning;
using namespace machine_learning::MLPACK_Classification;
using namespace machine_learning::MLPACK_Regression;
using namespace machine_learning::DLIB_Classification;
using namespace machine_learning::DLIB_Regression;


/*********************************************
	    Passive Aggressive Classifier
*********************************************/

PassiveAgressiveClassifier::PassiveAgressiveClassifier(string cfg, string net_name)
:MLPACK_Learner(){
	try{
		std::ifstream cfgfile(cfg);
		cfgfile >> root;
		regularization = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
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
			C = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
			    .get("C",-1).asDouble();
			if(C<0){
				cout << endl << "Incorrect parameter C." << endl;
				cout << endl << "Acceptable C parameters : [positive double]" << endl;
				throw;
			}
		}
		numberOfUpdates = 0;
	}catch(...){
		throw;
	}
}

void PassiveAgressiveClassifier::initializeModel(size_t sz){
	// Initialize the parameter vector W.
	W = arma::zeros<arma::dvec>(sz);
	intercept = 0.;
	//_model = arma::mat(&W(0), W.n_rows+1, 1, false);
	arma::mat* _model = new arma::mat(&W(0), W.n_rows+1, 1, false);
	vector_model.push_back(_model);
}

void PassiveAgressiveClassifier::update_model(const vector<arma::mat>& w){
	assert(w.size()==1);
	for(size_t i = 0; i < w.at(0).n_rows; i++){
		if(i<w.at(0).n_rows-1){
			W.row(i) = w.at(0).row(i);
		}else{
			intercept = w.at(0).row(i)(0);
		}
	}
}

void PassiveAgressiveClassifier::fit(const arma::mat& batch, const arma::mat& labels){
	// Starting the online learning
	for(size_t i=0; i<batch.n_cols; i++){
		
		// calculate the Hinge loss.
		double loss = 1. - labels(0,i)*(arma::dot(W,batch.unsafe_col(i))+intercept);
		
		// Classified correctly. Parameters stay the same. Passive approach.
		if (loss <= 0.){
			continue;
		}
			
		// Calculate the Lagrange Multiplier.
		double Lagrange_Muliplier = 0.;
		if(regularization=="none"){
			Lagrange_Muliplier = loss / ( arma::dot(batch.unsafe_col(i),batch.unsafe_col(i))+1 ) ;
		}else if (regularization=="l1"){
			Lagrange_Muliplier = std::min( C,loss / ( arma::dot(batch.unsafe_col(i),batch.unsafe_col(i))+1 ) );
		}else if (regularization=="l2"){
			Lagrange_Muliplier = loss / ( ( arma::dot(batch.unsafe_col(i),batch.unsafe_col(i))+1 ) + 1/(2*C) );
		}else{
			//throw exception
			cout << "throw exception" << endl;
		}
		
		// Update the parameters.
		W += Lagrange_Muliplier*labels(0,i)*batch.unsafe_col(i);
		intercept += Lagrange_Muliplier*labels(0,i);
	}
	numberOfUpdates += batch.n_cols;
}

arma::mat PassiveAgressiveClassifier::predict(const arma::mat& batch) const {
	arma::mat prediction = arma::zeros<arma::mat>(1,batch.n_cols);
	for(size_t i=0;i<batch.n_cols;i++){
		if( arma::dot(W,batch.unsafe_col(i)) + intercept >= 0. ){
			prediction(0,i) = 1.;
		}else{
			prediction(0,i) = -1.;
		}
	}
	return prediction;
}

inline vector<arma::mat*>& PassiveAgressiveClassifier::getModel(){
	(*vector_model.at(0))(vector_model.at(0)->n_rows-1, 0) = intercept;
	return vector_model;
}

inline vector<arma::SizeMat> PassiveAgressiveClassifier::modelDimensions() const{
	vector<arma::SizeMat> dims;
	dims.push_back(arma::size(*vector_model.at(0)));
	return dims; 
}

/*********************************************
	Extreme Learning Machine Classifier
*********************************************/

ELM_Classifier::ELM_Classifier(string cfg, string net_name)
:MLPACK_Learner(){
	try{
		std::ifstream cfgfile(cfg);
		cfgfile >> root;
		num_of_neurons = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
						.get("neurons",0).asInt();
		numberOfUpdates = 0;
	}catch(...){
		throw;
	}
}

void ELM_Classifier::initializeModel(size_t num_of_feats, size_t num_of_classes){
	A = 2.*arma::randu<arma::mat>(num_of_feats, num_of_neurons)-1.;
	b = 2.*arma::randu<arma::mat>(1, num_of_neurons)-1.;
	hidden_parameters.push_back(&A);
	hidden_parameters.push_back(&b);
	
	K = arma::zeros<arma::mat>(num_of_neurons, num_of_neurons);
	beta = arma::zeros<arma::mat>(num_of_neurons, num_of_classes);
	vector_model.push_back(&K);
	vector_model.push_back(&beta);
}

void ELM_Classifier::restoreModel(const vector<arma::mat*>& params){
	hidden_parameters.clear();
	A = arma::mat(arma::size(*params.at(0)),arma::fill::zeros);
	for(size_t i=0; i<params.at(0)->n_cols; i++){
		A.col(i) = params.at(0)->col(i);
	}
	b = arma::mat(arma::size(*params.at(1)),arma::fill::zeros);
	for(size_t i=0; i<params.at(1)->n_cols; i++){
		b.col(i) = params.at(1)->col(i);
	}
	hidden_parameters.push_back(&A);
	hidden_parameters.push_back(&b);
}

void ELM_Classifier::update_model(const vector<arma::mat>& w){
	vector_model.clear();
	// In case the learning has not been initialized.
	if(K.n_elem==0 || beta.n_elem==0){
		K = arma::mat(arma::size(w.at(0)), arma::fill::zeros);
		beta = arma::mat(arma::size(w.at(1)), arma::fill::zeros);
	}
		
    // In case of concept drift.
//	if(beta.n_cols != w.at(1).n_cols)
//		beta.resize(beta.n_rows, w.at(1).n_cols);
		
	// Update the parameters.
	for(size_t i = 0; i < w.at(0).n_cols; i++){
		K.col(i) = w.at(0).col(i);
	}
	for(size_t i = 0; i < w.at(1).n_cols; i++){
		beta.col(i) = w.at(1).col(i);
	}
	
	vector_model.push_back(&K);
	vector_model.push_back(&beta);
}

void ELM_Classifier::handleVD(size_t sz){
	vector_model.erase(vector_model.begin());
	arma::mat A_ = 2.*arma::randu<arma::mat>(A.n_rows+sz, num_of_neurons)-1.;
	A_.rows(0, A.n_rows-1) = A;
	A = A_;
	vector_model.insert(vector_model.begin(),&A);
}

void ELM_Classifier::handleRD(size_t sz){
	vector_model.erase(vector_model.begin()+vector_model.size()-1);
	arma::mat beta_ = arma::zeros<arma::mat>(num_of_neurons, beta.n_cols+sz);
	beta_.cols(0, beta.n_cols-1) = beta;
	beta = beta_;
	vector_model.push_back(&beta);
}

void ELM_Classifier::fit(const arma::mat& batch, const arma::mat& labels){
	// Check if the model has not been initialized.
	if(A.n_elem==0)
		initializeModel(batch.n_rows, labels.n_rows);
	
	// Check for Virtual Drift in the Stream. If True, expand the neurons of the network.
	if(batch.n_rows>A.n_rows)
		handleVD(batch.n_rows-A.n_rows);
		
	// Check for Real Drift in the Stream. If True, expand the beta parameters of the network.
	if(labels.n_rows>beta.n_cols)
		handleRD(labels.n_rows-beta.n_cols);
	
	// Fit the data to the network.
	arma::mat H = batch.t()*A;
	H = arma::tanh(H.each_row()+b);
	arma::mat H_T = H.t();
	K += H_T*H;
	beta += arma::inv(K)*H_T*(labels.t()-H*beta);
	
	numberOfUpdates += batch.n_cols;
}

arma::mat ELM_Classifier::predict(const arma::mat& batch) const{
	
	arma::mat H = batch.t()*A;
	H = arma::tanh(H.each_row()+b);
	arma::mat predictionTemp = H*beta;
	
	arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_rows);
	for(size_t i=0; i<predictionTemp.n_rows; i++){
		prediction(0,i) = arma::as_scalar( arma::find( arma::max(predictionTemp.row(i)) == predictionTemp.row(i), 1) );
	}
	
	return prediction;
}

inline double ELM_Classifier::accuracy(const arma::mat& testbatch, const arma::mat& labels) const{
	// Calculate accuracy.
	int errors=0;
	arma::mat prediction = predict(testbatch);
	for(size_t i=0; i<labels.n_cols; i++){
		if(prediction(0,i) != arma::as_scalar( arma::find( arma::max(labels.unsafe_col(i)) == labels.unsafe_col(i), 1) ))
			errors++;
	}
	double score = 100.0*(labels.n_elem-errors)/(labels.n_elem);
	return score;
}

inline vector<arma::mat*>& ELM_Classifier::getHModel(){
	return hidden_parameters;
}

inline vector<arma::SizeMat> ELM_Classifier::modelDimensions() const{
	vector<arma::SizeMat> md_size;
	md_size.push_back(arma::size(*vector_model.at(0)));
	md_size.push_back(arma::size(*vector_model.at(1)));
	return md_size; 
}

/*********************************************
	Multi Layer Perceptron Classifier
*********************************************/
/*
MLP_Classifier::MLP_Classifier(string cfg, string net_name)
:MLPACK_Learner(){
	try{
		
		std::ifstream cfgfile(cfg);
		cfgfile >> root;
		
		size_t num_hidden_lrs = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
							    .get("Number_of_hidden_layers",-1).asInt();
		if(num_hidden_lrs <= 0){
			cout << endl <<"Invalid parameter number of hidden layers." << endl;
			cout << "Parameter Number_of_hidden_layers must be a positive integer." << endl;
			throw;
		}
		
		stepSize = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
				   .get("stepSize",-1).asDouble();
		if(stepSize <= 0.){
			cout << endl <<"Invalid parameter stepSize given." << endl;
			cout << "Parameter stepSize must be a positive double." << endl;
			throw;
		}
		
		batch_size = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
				   .get("batchSize",-1).asInt64();
		if(batch_size <= 0.){
			cout << endl <<"Invalid parameter batchSize given." << endl;
			cout << "Parameter batchSize must be a positive integer." << endl;
			throw;
		}
		
		beta1 = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
				.get("beta1",-1).asDouble();
		if(beta1 <= 0.){
			cout << endl <<"Invalid parameter beta1 given." << endl;
			cout << "Parameter beta1 must be a positive double." << endl;
			throw;
		}
		
		beta2 = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
				.get("beta2",-1).asDouble();
		if(beta2 <= 0.){
			cout << endl <<"Invalid parameter beta2 given." << endl;
			cout << "Parameter beta2 must be a positive double." << endl;
			throw;
		}
		
		eps = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
			  .get("eps",-1).asDouble();
		if(eps <= 0.){
			cout << endl <<"Invalid parameter eps given." << endl;
			cout << "Parameter eps must be a positive double." << endl;
			throw;
		}
		
		maxIterations = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
						.get("maxIterations",-1).asInt();
		if(maxIterations <= 0 ){
			cout << endl <<"Invalid parameter maxIterations given." << endl;
			cout << "Parameter maxIterations must be a positive integer." << endl;
			throw;
		}
		
		tolerance = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
					.get("tolerance",-1).asDouble();
		if(tolerance <= 0.){
			cout << endl <<"Invalid parameter tolerance given." << endl;
			cout << "Parameter tolerance must be a positive double." << endl;
			throw;
		}
		
		for(size_t i = 0; i < num_hidden_lrs; ++i){
			
			layer_size.push_back(root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
								 .get("hidden"+std::to_string(i+1),-1).asInt());
			if(layer_size.back()<=0){
				cout << endl << "Invalid size of hidden layer " << i+1 << "." << endl;
				cout << "Size of hidden layer must be a positive integer." << endl;
				throw;
			}
			
			layer_activation.push_back(root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
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
		numberOfUpdates = 0;
	}catch(...){
		throw;
	}
}

void MLP_Classifier::initializeModel(size_t sz){
	// Built the model
	model = new FFN<NegativeLogLikelihood<>, GaussianInitialization>();
	
	for(unsigned i=0;i<layer_size.size();i++){
		
		if(i==0){
			model->Add<Linear<> >(sz,layer_size.at(i));
		}else{
			model->Add<Linear<> >(layer_size.at(i-1),layer_size.at(i));
		}
		
		if(layer_activation.at(i) == "logistic"){
			model->Add<SigmoidLayer<>>();
		}else if(layer_activation.at(i) == "tanh"){
			model->Add<TanHLayer<>>();
		}else if(layer_activation.at(i) == "relu"){
			model->Add<ReLULayer<>>();
		}else{
			cout << endl << "No support for this activation function at the moment." << endl;
			throw;
		}
		
	}
	
	model->Add<Linear<> >(layer_size.at(layer_size.size()-1), 2);
	model->Add<LogSoftMax<> >();
	model->ResetParameters();
	
	opt = new SGD<AdamUpdate>(stepSize, batch_size, maxIterations, tolerance);
	opt->UpdatePolicy() = AdamUpdate(eps, beta1, beta2);
	opt->ResetPolicy() = false;
	opt->UpdatePolicy().Initialize(model->Parameters().n_rows, model->Parameters().n_cols);
	
	arma::mat* _model = new arma::mat(&model->Parameters()(0,0), model->Parameters().n_rows, model->Parameters().n_cols, false);
	vector_model.push_back(_model);
}

void MLP_Classifier::update_model(const vector<arma::mat>& w){
	assert(w.size()==1);
	for(size_t i = 0; i < w.at(0).n_cols; i++){
		model->Parameters().col(i) = w.at(0).col(i);
	}
}

void MLP_Classifier::fit(const arma::mat& batch, const arma::mat& labels){
	model->Train( batch, labels, *opt );
	numberOfUpdates+=batch.n_cols;
}

arma::mat MLP_Classifier::predict(const arma::mat& batch) const {
	arma::mat predictionTemp;
	model->Predict(batch, predictionTemp);
	
	arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);
	for(size_t i = 0; i < predictionTemp.n_cols; ++i){
		prediction(0,i) = arma::as_scalar( arma::find( arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1) );
	}
	
	return prediction;
}

inline double MLP_Classifier::accuracy(const arma::mat& testbatch, const arma::mat& labels) const {
	arma::mat predictionTemp;
	model->Predict(testbatch, predictionTemp);
	
	arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);
	for(size_t i = 0; i < predictionTemp.n_cols; ++i){
		prediction(0,i) = arma::as_scalar( arma::find( arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1) )+1;
	}
	
	int errors = 0;
	for(unsigned i = 0; i < prediction.n_cols; ++i){
		if(labels(0,i) != prediction(0,i)){
			errors++;
		}
	}
	
	return (double)errors;
}

inline vector<arma::SizeMat> MLP_Classifier::modelDimensions() const{
	vector<arma::SizeMat> md_size;
	for(arma::mat* param:vector_model)
		md_size.push_back(arma::size(*param));
	return md_size; 
}
*/

/*********************************************
	Passive Aggressive Regressor
*********************************************/

PassiveAgressiveRegression::PassiveAgressiveRegression(string cfg, string net_name)
:MLPACK_Learner(){
	try{
		std::ifstream cfgfile(cfg);
		cfgfile >> root;
		
		regularization = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
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
			C = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
			    .get("C",-1).asDouble();
			if(C<0){
				cout << endl << "Incorrect parameter C." << endl;
				cout << endl << "Acceptable C parameters : [positive double]" << endl;
				throw;
			}
		}
		
		epsilon = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
				  .get("epsilon",0.1).asDouble();
		if(epsilon<=0.){
			cout << endl << "Invalid parameter epsilon." << endl;
			cout << "Parameter epsilon must be a positive double." << endl;
			throw;
		}
		numberOfUpdates = 0;
	}catch(...){
		throw;
	}
}

void PassiveAgressiveRegression::initializeModel(size_t sz){
	// Initialize the parameter vector W.
	W = arma::zeros<arma::dvec>(sz);
	arma::mat* _model = new arma::mat(&W(0), W.n_rows, 1, false);
	vector_model.push_back(_model);
}

void PassiveAgressiveRegression::update_model(const vector<arma::mat>& w){
	assert(w.size()==1);
	W -= W - w.at(0).unsafe_col(0);
}

void PassiveAgressiveRegression::fit(const arma::mat& batch, const arma::mat& labels){
	// Starting the online learning
	for(size_t i=0; i<batch.n_cols; i++){
		
		// calculate the epsilon-insensitive loss.
		double pred = arma::dot(W,batch.unsafe_col(i) );
		double loss = std::max( 0.0 , std::abs( labels(0,i) - pred) - epsilon );
			
		// Calculate the Lagrange Multiplier.
		double Lagrange_Muliplier = 0.;
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
		
		double sign = 1.0;
		if (labels(0,i) - pred < 0)
			sign = -1.0;
		
		// Update the parameters.
		W += Lagrange_Muliplier*sign*batch.unsafe_col(i);
	}
	numberOfUpdates+=batch.n_cols;
}

arma::mat PassiveAgressiveRegression::predict(const arma::mat& batch) const {
	arma::mat prediction = arma::zeros<arma::mat>(1,batch.n_cols);
	for(size_t i=0;i<batch.n_cols;i++){
		prediction(0,i) = arma::dot(W,batch.unsafe_col(i));
	}
	return prediction;
}

inline vector<arma::SizeMat> PassiveAgressiveRegression::modelDimensions() const {
	vector<arma::SizeMat> md_size;
	for(arma::mat* param:vector_model)
		md_size.push_back(arma::size(*param));
	return md_size;
}

/*********************************************
	Neural Network Regressor
*********************************************/

/*
NN_Regressor::NN_Regressor(string cfg, string net_name)
:MLPACK_Learner(){
	try{
		std::ifstream cfgfile(cfg);
		cfgfile >> root;
		
		int sz_input_layer = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
							 .get("Size_of_input_layer",-1).asInt(); 
		if(sz_input_layer <= 0){
			cout << endl <<"Invalid size for input layers." << endl;
			cout << "Parameter Size_of_input_layer must be a positive integer." << endl;
			throw;
		}
		
		size_t num_hidden_lrs = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
							    .get("Number_of_hidden_layers",-1).asInt();
		if(num_hidden_lrs <= 0){
			cout << endl <<"Invalid parameter number of hidden layers." << endl;
			cout << "Parameter Number_of_hidden_layers must be a positive integer." << endl;
			throw;
		}
		
		stepSize = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
				   .get("stepSize",-1).asDouble();
		if(stepSize <= 0.){
			cout << endl <<"Invalid parameter stepSize given." << endl;
			cout << "Parameter stepSize must be a positive double." << endl;
			throw;
		}
		
		batch_size = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
				   .get("batchSize",-1).asInt64();
		if(batch_size <= 0.){
			cout << endl <<"Invalid parameter batchSize given." << endl;
			cout << "Parameter batchSize must be a positive integer." << endl;
			throw;
		}
		
		beta1 = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
				.get("beta1",-1).asDouble();
		if(beta1 <= 0.){
			cout << endl <<"Invalid parameter beta1 given." << endl;
			cout << "Parameter beta1 must be a positive double." << endl;
			throw;
		}
		
		beta2 = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
				.get("beta2",-1).asDouble();
		if(beta2 <= 0.){
			cout << endl <<"Invalid parameter beta2 given." << endl;
			cout << "Parameter beta2 must be a positive double." << endl;
			throw;
		}
		
		eps = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
			  .get("eps",-1).asDouble();
		if(eps <= 0.){
			cout << endl <<"Invalid parameter eps given." << endl;
			cout << "Parameter eps must be a positive double." << endl;
			throw;
		}
		
		maxIterations = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
						.get("maxIterations",-1).asInt();
		if(maxIterations <= 0 ){
			cout << endl <<"Invalid parameter maxIterations given." << endl;
			cout << "Parameter maxIterations must be a positive integer." << endl;
			throw;
		}
		
		tolerance = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
					.get("tolerance",-1).asDouble();
		if(tolerance <= 0.){
			cout << endl <<"Invalid parameter tolerance given." << endl;
			cout << "Parameter tolerance must be a positive double." << endl;
			throw;
		}
		
		for(size_t i = 0; i < num_hidden_lrs; ++i){
			
			layer_size.push_back(root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
								 .get("hidden"+std::to_string(i+1),-1).asInt());
			if(layer_size.back()<=0){
				cout << endl << "Invalid size of hidden layer " << i+1 << "." << endl;
				cout << "Size of hidden layer must be a positive integer." << endl;
				throw;
			}
			
			layer_activation.push_back(root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
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
		numberOfUpdates = 0;
	}catch(...){
		throw;
	}
}

void NN_Regressor::initializeModel(size_t sz){
	// Built the model
	model = new FFN<MeanSquaredError<>, GaussianInitialization>();
	for(unsigned i=0;i<layer_size.size();i++){
		if(i==0){
			model->Add<Linear<> >(sz,layer_size.at(i));
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
	model->Add<Linear<> >(layer_size.at(layer_size.size()-1), 1);
	model->ResetParameters();
	
	opt = new SGD<AdamUpdate>(stepSize, batch_size, maxIterations, tolerance);
	opt->UpdatePolicy() = AdamUpdate(eps, beta1, beta2);
	opt->ResetPolicy() = false;
	opt->UpdatePolicy().Initialize(model->Parameters().n_rows, model->Parameters().n_cols);
	
	arma::mat* _model = new arma::mat(&model->Parameters()(0,0), model->Parameters().n_rows, model->Parameters().n_cols, false);
	vector_model.push_back(_model);
}

void NN_Regressor::update_model(const vector<arma::mat>& w){
	assert(w.size()==1);
	for(size_t i = 0; i < w.at(0).n_cols; i++){
		model->Parameters().col(i) = w.at(0).col(i);
	}
}

void NN_Regressor::fit(const arma::mat& batch, const arma::mat& labels){
	model->Train( batch, labels, *opt );
	numberOfUpdates+=batch.n_cols;
}

arma::mat NN_Regressor::predict(const arma::mat& batch) const {
	arma::mat prediction;
	model->Predict(batch, prediction);
	return prediction;
}

inline double NN_Regressor::accuracy(const arma::mat& test_data, const arma::mat& labels) const {
	arma::mat prediction;
	model->Predict(test_data, prediction);
	
	// Calculate accuracy RMSE.
	double RMSE = 0;
	for(size_t i=0;i<labels.n_cols;i++){
		RMSE += std::pow( labels(0,i) - prediction(0,i) , 2);
	}
	cout << endl << "(RMSE*T)^2 = " << RMSE << endl;
	RMSE /= labels.n_cols;
	
	return std::sqrt(RMSE);
}

inline vector<arma::SizeMat> NN_Regressor::modelDimensions() const {
	vector<arma::SizeMat> md_size;
	for(arma::mat* param:vector_model)
		md_size.push_back(arma::size(*param));
	return md_size; 
}

*/