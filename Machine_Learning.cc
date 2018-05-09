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
	_model = arma::mat(&W(0), W.n_rows+1, 1, false);
}

void PassiveAgressiveClassifier::update_model(arma::mat w){
	for(size_t i = 0; i < w.n_rows; i++){
		if(i<w.n_rows-1){
			W.row(i) = w.row(i);
		}else{
			intercept = w.row(i)(0);
		}
	}
}

void PassiveAgressiveClassifier::update_model_by_ref(const arma::mat& w){
	//W -= W - w.unsafe_col(0);
	for(size_t i = 0; i < w.n_rows; i++){
		if(i<w.n_rows-1){
			W.row(i) = w.row(i);
		}else{
			intercept = w.row(i)(0);
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
		double Lagrange_Muliplier;
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
	numberOfUpdates+=batch.n_cols;
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


/*********************************************
	Kernel Passive Aggressive Classifier
*********************************************/

KernelPassiveAgressiveClassifier::KernelPassiveAgressiveClassifier(string cfg, string net_name)
:MLPACK_Learner() {
	try{
		
		std::ifstream cfgfile(cfg);
		cfgfile >> root;
		
		time_seed = (long int)root[root["gm_network_"+net_name]
							   .get("parameters_of_learner","NoParams").asString()]
		                       .get("seed",0).asInt64();
		if(time_seed < 0){
			seed = time(&time_seed);
			std::srand (seed);
		}else{
			std::srand (time_seed);
		}
		
		kernel = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
					 .get("kernel","No Kernel given").asString();
		if(kernel!="poly" && kernel!="rbf"){
			cout << endl << "Invalid kernel given." << endl;
			cout << "Valid kernels [\"rbf\",\"poly\"] ." << endl;
			throw;
		}
		if(kernel=="rbf"){
			double gamma = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
						   .get("gamma",-1).asDouble();
			if(gamma<=0.){
				cout << endl << "Invalid gamma given." << endl;
				cout << "Sigma must be a positive double." << endl;
				cout << "Sigma is set to 0.5 by default" << endl;
				kernel_params.push_back(0.5);
			}else{
				kernel_params.push_back(gamma);
			}
		}else if(kernel=="poly"){
			int degree = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
						 .get("degree",-1).asInt();
			double offset = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
			                .get("offset",1e-20).asDouble();
			if(degree<=0 && offset!=1e-20){
				cout << endl << "Invalid degree given." << endl;
				cout << "Degree must be a positive integer." << endl;
				cout << "Sigma is set to 2.0 by default" << endl;
				kernel_params.push_back(2);
				kernel_params.push_back(offset);
			}else if(degree>0 && offset==1e-20){
				cout << endl << "Offset id 0.0 by default" << endl;
				kernel_params.push_back(degree);
				kernel_params.push_back(0.);
			}else if(degree<=0 && offset==1e-20){
				cout << endl << "Invalid degree given." << endl;
				cout << "Degree must be a positive integer." << endl;
				cout << "Sigma is set to 2.0 by default" << endl;
				cout << endl << "Offset id 0.0 by default" << endl;
				kernel_params.push_back(2);
				kernel_params.push_back(0.);
			}else{
				kernel_params.push_back(degree);
				kernel_params.push_back(offset);
			}
		}
		
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
		
		double alpha = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
					   .get("a",-1.).asDouble();
		double beta = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
		              .get("b",-1.).asDouble();
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
		
		int mSVs = root[root["gm_network_"+net_name].get("parameters_of_learner","NoParams").asString()]
		           .get("maxSVs",-1).asInt64();
		if(mSVs<=0){
			cout << endl << "Invalid parameter maxSVs." << endl;
			cout << "The parameter maxSVs must be a positive integer." << endl;
			throw;
		}
		maxSVs = mSVs;
		numberOfUpdates = 0;
	}catch(...){
		throw;
	}
}

void KernelPassiveAgressiveClassifier::initializeModel(size_t sz) {
	// Initialize the parameter vector W.
	_model = arma::zeros<arma::mat>(maxSVs,sz+2);
}

void KernelPassiveAgressiveClassifier::update_model(arma::mat w) {
	_model = w; // Update the model.
	// Create the red-black tree with the new Support Vectors.
	if(position_set.size() != 0)
		position_set.clear();
	for(size_t i = 0; i < _model.n_rows; i++){
		if(_model(i,0)!=0.){ // If not true then it means that the num of SVs are still less than the max allowed
			position_set.insert(SV_index(i,std::abs(_model(i,1))/_model(i,0)));
		}else{
			break;
		}
	}
	updateRefModel();
}

void KernelPassiveAgressiveClassifier::update_model_by_ref(const arma::mat& w) {
	// Create the red-black tree with the new Support Vectors.
	if(position_set.size() != 0)
		position_set.clear();
	for(size_t i = 0; i < _model.n_rows; i++){
		_model.row(i) = w.row(i);
		if(_model(i,0)!=0.){ // If not true then it means that the num of SVs are still less than the max allowed
			position_set.insert(SV_index(i,std::abs(_model(i,1))/_model(i,0)));
		}
	}
	updateRefModel();
}

void KernelPassiveAgressiveClassifier::fit(const arma::mat& batch, const arma::mat& labels) {
	// Starting the online learning
	for(size_t i=0; i<batch.n_cols; i++){
		arma::dvec point = batch.unsafe_col(i);
		
		double loss;
		double probability;
		if(position_set.size()==0){
			loss = (double)std::rand()/RAND_MAX;
		}else{
			// calculate the Hinge loss.
			loss = 1. - labels(0,i)*this->Margin(point);
		}
		
		// Classified correctly. Parameters stay the same. Passive approach.
		if (loss <= 0.){
			continue;
		}
			
		probability = std::min(a,loss)/b;
		//probability = 1.;
		if ( (double)std::rand()/RAND_MAX <= probability || position_set.size()==0 ){
			
			// Calculate the Lagrange Multiplier.
			double Lagrange_Multiplier;
			if(regularization=="none"){
				Lagrange_Multiplier = loss / HilbertDot(point,point);
			}else if (regularization=="l1"){
				Lagrange_Multiplier = std::min( C/probability, loss/HilbertDot(point,point) );
			}else if (regularization=="l2"){
				Lagrange_Multiplier = loss / ( HilbertDot(point,point) + probability/(2*C) );
			}else{
				//throw exception
				cout << "throw exception" << endl;
			}
			
			// Keep only the best Support Vectors.
			if(position_set.size()<maxSVs){
				std::pair pr = position_set.insert(SV_index(position_set.size(), Lagrange_Multiplier));
				if(!pr.second)
					continue;
				_model(position_set.size()-1,0) = 1.; // Multiplied by...
				_model(position_set.size()-1,1) = Lagrange_Multiplier*labels(0,i); // The multiplier.
				_model.row(position_set.size()-1).cols(2, _model.n_cols-1) = batch.col(i).t(); // The SV.
				updateRefModel();
				continue;
			}
			if(position_set.begin()->La_Mult < Lagrange_Multiplier){
				// Insert the new row vector in the ordered set.
				std::pair pr = position_set.insert(SV_index(position_set.begin()->index, Lagrange_Multiplier));
				if(!pr.second)
					continue;
				_model(position_set.begin()->index,0) = 1.; // Multiplied by...
				_model(position_set.begin()->index,1) = Lagrange_Multiplier*labels(0,i); // The multiplier.
				_model.row(position_set.begin()->index).cols(2, _model.n_cols-1) = batch.col(i).t(); // The SV.
				position_set.erase(position_set.begin()); // erase the first(smallest) element from the set.
				updateRefModel();
			}
		}
	}
	numberOfUpdates+=batch.n_cols;
}

void KernelPassiveAgressiveClassifier::updateRefModel() {
	if( (position_set.size()<maxSVs) || (position_set.size()==maxSVs && coefs.n_rows<maxSVs) ){
		coefs = arma::mat(&_model(0,1), position_set.size(), 1, false);
		SVs = arma::mat(&_model(0,2), _model.n_rows, _model.n_cols-2, false);
		if(position_set.size()<maxSVs)
			SVs.shed_rows(position_set.size(), _model.n_rows-1);
	}
	for(size_t i = 0; i< coefs.n_rows; i++){
		assert(_model(i,1)==coefs(i,0));
		for(size_t j = 0; j< SVs.n_cols; j++){
			assert(_model(i,j+2)==SVs(i,j));
		}
	}
}

arma::mat KernelPassiveAgressiveClassifier::predict(const arma::mat& batch) const {
	if(kernel=="poly"){
		arma::mat predictions = coefs.t()*arma::pow( SVs*batch + kernel_params.at(1), (int)kernel_params.at(0));
		predictions.transform([](double val){ return (val >= 0.) ? 1.: -1.; } );
		return predictions;
	}else{
		arma::mat prediction = arma::zeros<arma::mat>(1,batch.n_cols);
		for(size_t i = 0; i < batch.n_cols; i++){
			arma::dvec point = batch.unsafe_col(i);
			if( this->Margin(point) >= 0.){
				prediction(0,i) = 1.;
			}else{
				prediction(0,i) = -1.;
			}
		}
		return prediction;
	}
}

double KernelPassiveAgressiveClassifier::Margin(arma::dvec& data_point) const {
	if(kernel=="poly"){
		return arma::dot(coefs, arma::pow( SVs*data_point + kernel_params.at(1), (int)kernel_params.at(0)));
	}else{
		return arma::dot(coefs,
						 arma::exp(-kernel_params.at(0)*
								   (SVs.each_row()-data_point.t())
								   .each_row( [](arma::drowvec& a){ a(0)=arma::dot(a,a);return a; })
								   .col(0) 
								   )
					    );
	}
}

double KernelPassiveAgressiveClassifier::HilbertDot(const arma::dvec& data_point1, const arma::dvec& data_point2) const {
	if(kernel=="poly"){ // Hilbert Product for Polynomial Kernel 
		return std::pow(arma::dot(data_point1, data_point2) + kernel_params.at(1), (int)kernel_params.at(0));
	}else{ // Hilbert Product for Gaussian Kernel 
		return 1.;
	}
}


/*********************************************
	Multi Layer Perceptron Classifier
*********************************************/

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
	
	//opt = new SGD<AdamUpdate>(stepSize, batch_size, beta1, beta2, eps, maxIterations, tolerance, false);
	opt = new SGD<AdamUpdate>(stepSize, batch_size, maxIterations, tolerance);
	opt->UpdatePolicy() = AdamUpdate(eps, beta1, beta2);
	opt->ResetPolicy() = false;
	opt->UpdatePolicy().Initialize(model->Parameters().n_rows, model->Parameters().n_cols);
	
	_model = arma::mat(&model->Parameters()(0,0), model->Parameters().n_rows, model->Parameters().n_cols, false);

}

void MLP_Classifier::update_model(arma::mat w){
	model->Parameters() = w;
}

void MLP_Classifier::update_model_by_ref(const arma::mat& w){
	for(size_t i = 0; i < w.n_cols; i++){
		model->Parameters().col(i) = w.col(i);
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
	_model = arma::mat(&W(0), W.n_rows, 1, false);
}

void PassiveAgressiveRegression::update_model(arma::mat w){
	W = w.col(0);
}

void PassiveAgressiveRegression::update_model_by_ref(const arma::mat& w){
	W -= W - w.unsafe_col(0);
}

void PassiveAgressiveRegression::fit(const arma::mat& batch, const arma::mat& labels){
	// Starting the online learning
	for(size_t i=0; i<batch.n_cols; i++){
		
		// calculate the epsilon-insensitive loss.
		double pred = arma::dot(W,batch.unsafe_col(i) );
		double loss = std::max( 0.0 , std::abs( labels(0,i) - pred) - epsilon );
			
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


/*********************************************
	Neural Network Regressor
*********************************************/

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
		opt = new Adam(stepSize, 1, beta1, beta2, eps, maxIterations, tolerance, false);
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
}

void NN_Regressor::update_model(arma::mat w){
	model->Parameters() = w;
}

void NN_Regressor::update_model_by_ref(const arma::mat& w){
	model->Parameters() -= model->Parameters() - w;
}

void NN_Regressor::fit(const arma::mat& batch, const arma::mat& labels){
	if(batch.n_cols != opt->BatchSize()){
		delete opt;
		opt = new Adam( stepSize, batch.n_cols, beta1, beta2, eps, maxIterations, tolerance, false );
	}
	model->Train( batch, labels, *opt );
	numberOfUpdates+=batch.n_cols;
}

arma::mat NN_Regressor::predict(const arma::mat& batch) const {
	arma::mat prediction;
	model->Predict(batch, prediction);
	return prediction;
}


