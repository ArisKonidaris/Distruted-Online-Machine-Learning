#include "ML_GM_Proto.hh"

using namespace ml_gm_proto;
using namespace ml_gm_proto::MlPack_GM_Proto;
using namespace ml_gm_proto::DLib_GM_Proto;


/*********************************************
	ml_safezone_function
*********************************************/

ml_safezone_function::ml_safezone_function(arma::mat& mdl)
:GlobalModel(mdl)
{ }

ml_safezone_function::~ml_safezone_function() { }

const arma::mat& ml_safezone_function::getGlobalModel() const {
	return GlobalModel;
}


/*********************************************
	Batch_Learning
*********************************************/

Batch_Learning::Batch_Learning(arma::mat& GlMd)
: ml_safezone_function(GlMd), threshold(32)
{ }

Batch_Learning::Batch_Learning(arma::mat& GlMd, size_t thr)
: ml_safezone_function(GlMd), threshold(thr)
{ }

double Batch_Learning::checkIfAdmissible(const size_t counter) const
{ 
	return (double)(threshold-counter);
}

Batch_Learning::~Batch_Learning() { }


/*********************************************
	Variance_safezone_func
*********************************************/

Variance_safezone_func::Variance_safezone_func(arma::mat& GlMd)
: ml_safezone_function(GlMd), threshold(1.), batch_size(32)
{ }

Variance_safezone_func::Variance_safezone_func(arma::mat& GlMd, size_t batch_sz)
: ml_safezone_function(GlMd), threshold(1.), batch_size(batch_sz)
{ }

Variance_safezone_func::Variance_safezone_func(arma::mat& GlMd, double thr)
: ml_safezone_function(GlMd), threshold(thr), batch_size(32)
{ }

Variance_safezone_func::Variance_safezone_func(arma::mat& GlMd, double thr, size_t batch_sz)
: ml_safezone_function(GlMd), threshold(thr), batch_size(batch_sz)
{ }

double Variance_safezone_func::checkIfAdmissible(const size_t counter) const
{ 
	double sz = (double)(batch_size-counter);
	return sz; 
}

double Variance_safezone_func::checkIfAdmissible(const arma::mat& mdl) const
{
	return threshold - std::pow(arma::norm(mdl-GlobalModel,2),2);
}

Variance_safezone_func::~Variance_safezone_func() { }


/*********************************************
	safezone
*********************************************/

safezone::safezone() 
: szone(nullptr) { }

safezone::~safezone() { }

// valid safezone
safezone::safezone(ml_safezone_function* sz)
: szone(sz) { }

// Movable
safezone::safezone(safezone&& other)
{
	swap(other);
}

safezone& safezone::operator=(safezone&& other)
{
	swap(other);
	return *this;
}

// Copyable
safezone::safezone(const safezone& other){
	szone = other.szone;
}

safezone& safezone::operator=(const safezone& other){
	if(szone != other.szone) {
		szone = other.szone;
	}
	return *this;
}


/*********************************************
	dl_query_state
*********************************************/

query_state::query_state()
{ }

query_state::query_state(arma::SizeMat sz)
:GlobalModel(arma::mat(sz, arma::fill::zeros))
{ }

query_state::~query_state() { }

ml_safezone_function* query_state::safezone(string cfg, string algo){
	
	Json::Value root;
	std::ifstream cfgfl(cfg);
	cfgfl >> root;
	
	string algorithm = root[algo].get("algorithm","Batch_Learning").asString();
	if(algorithm == "Batch_Learning"){
		auto safe_zone = new Batch_Learning(GlobalModel, root[algo].get("batch_size",32).asInt64());
		return safe_zone;
	}else if(algorithm == "Michael_Kamp"){
		auto safe_zone = new Variance_safezone_func(GlobalModel,
		                                            root[algo].get("threshold",1.).asDouble(),
													root[algo].get("batch_size",32).asInt64());
		return safe_zone;
	}else if(algorithm == "Michael_Kamp_Kernel"){
		auto safe_zone = new Batch_Learning(GlobalModel, root[algo].get("batch_size",32).asInt64());
		return safe_zone; 
	}else{
		return nullptr;
	}	
}


/*********************************************
	continuous_query
*********************************************/

continuous_query::continuous_query(arma::mat* tSet, arma::mat* tRes, string cfg, string nm)
:testSet(tSet),testResponses(tRes)
{
	cout << "Initializing the query..." << endl;
	Json::Value root;
	std::ifstream cfgfl(cfg);
	cfgfl >> root;
	
	config.learning_algorithm = root["gm_network_"+nm]
						        .get("learning_algorithm", "Trash").asString();
	config.distributed_learning_algorithm = root["gm_network_"+nm]
									        .get("distributed_learning_algorithm", "Trash").asString();
	config.network_name = nm;
	config.cfgfile = cfg;
	
	cout << "Query initialized : " << config.learning_algorithm << ", ";
	cout << config.distributed_learning_algorithm << ", ";
	cout << config.network_name << ", ";
	cout << config.cfgfile << endl;
}


/*********************************************
	Classification_query
*********************************************/

Classification_query::Classification_query(arma::mat* tSet, arma::mat* tRes, string cfg, string nm)
:continuous_query(tSet,tRes,cfg,nm) { }

double Classification_query::queryAccuracy(MLPACK_Learner* lnr) {
	if(config.learning_algorithm == "MLP"){
		testResponses->transform( [](double val) { return (val == -1.) ? 1.: val+1.; } );
		double accuracy = 100.0*(testResponses->n_cols-lnr->accuracy(*testSet,*testResponses))/(testResponses->n_cols);
		testResponses->transform( [](double val) { return (val == 1.) ? -1.: val-1; } );
		return accuracy;
	}else if(config.learning_algorithm == "PA" || config.learning_algorithm == "Kernel_PA"){
		return 100.0*(testResponses->n_cols-lnr->accuracy(*testSet,*testResponses))/(testResponses->n_cols);
	}else{
		return lnr->accuracy(*testSet,*testResponses);
	}
}


/*********************************************
	Regression_query
*********************************************/

Regression_query::Regression_query(arma::mat* tSet, arma::mat* tRes, string cfg, string nm)
:continuous_query(tSet,tRes,cfg,nm) { }

double Regression_query::queryAccuracy(MLPACK_Learner* lnr) {
	return lnr->accuracy(*testSet,*testResponses);
}


/*********************************************
	dl_safezone_function
*********************************************/

/*
dl_safezone_function::dl_safezone_function(resizable_tensor& mdl)
:GlobalModel(mdl)
{ }
*/

dl_safezone_function::dl_safezone_function(vector<resizable_tensor*>& mdl)
:GlobalModel(mdl)
{
	num_of_params=0;
	for(auto layer:mdl){
		num_of_params+=layer->size();
	}
}

/*
dl_safezone_function::dl_safezone_function(vector<tensor*>& mdl)
{
	GlobalModel.clear();
	num_of_params=0;
	for(auto layer:mdl){
		resizable_tensor l(*layer);
		GlobalModel.push_back(&l);
		num_of_params+=layer->size();
	}
}
*/

dl_safezone_function::~dl_safezone_function() { }

const vector<resizable_tensor*>& dl_safezone_function::getGlobalModel() const {
	return GlobalModel;
}


/*********************************************
	Batch_safezone_function
*********************************************/
//Batch_safezone_function::Batch_safezone_function(resizable_tensor& GlMd)
Batch_safezone_function::Batch_safezone_function(vector<resizable_tensor*>& GlMd)
: dl_safezone_function(GlMd), threshold(32){ 
	hyperparameters.push_back((float)32);
}

//Batch_safezone_function::Batch_safezone_function(resizable_tensor& GlMd, size_t thr)
Batch_safezone_function::Batch_safezone_function(vector<resizable_tensor*>& GlMd, size_t thr)
: dl_safezone_function(GlMd), threshold(thr){
	hyperparameters.push_back((float)thr);
}

double Batch_safezone_function::checkIfAdmissible(const size_t counter) const
{ 
	return (double)(threshold-counter);
}

Batch_safezone_function::~Batch_safezone_function() { }


/*********************************************
	Param_Variance_safezone_func
*********************************************/

//Param_Variance_safezone_func::Param_Variance_safezone_func(resizable_tensor& GlMd)
Param_Variance_safezone_func::Param_Variance_safezone_func(vector<resizable_tensor*>& GlMd)
: dl_safezone_function(GlMd), threshold(1.), batch_size(32)
{ 
	num_of_params=0;
	for(auto layer:GlMd){
		num_of_params+=layer->size();
	}
	hyperparameters.push_back((float)1.);
	hyperparameters.push_back((float)32);
}

//Param_Variance_safezone_func::Param_Variance_safezone_func(resizable_tensor& GlMd, size_t batch_sz)
Param_Variance_safezone_func::Param_Variance_safezone_func(vector<resizable_tensor*>& GlMd, size_t batch_sz)
: dl_safezone_function(GlMd), threshold(1.), batch_size(batch_sz)
{ 
	num_of_params=0;
	for(auto layer:GlMd){
		num_of_params+=layer->size();
	}
	hyperparameters.push_back((float)1.);
	hyperparameters.push_back((float)batch_sz);
}

//Param_Variance_safezone_func::Param_Variance_safezone_func(resizable_tensor& GlMd, double thr)
Param_Variance_safezone_func::Param_Variance_safezone_func(vector<resizable_tensor*>& GlMd, double thr)
: dl_safezone_function(GlMd), threshold(thr), batch_size(32)
{ 
	num_of_params=0;
	for(auto layer:GlMd){
		num_of_params+=layer->size();
	}
	hyperparameters.push_back((float)thr);
	hyperparameters.push_back((float)32);
}

//Param_Variance_safezone_func::Param_Variance_safezone_func(resizable_tensor& GlMd, double thr, size_t batch_sz)
Param_Variance_safezone_func::Param_Variance_safezone_func(vector<resizable_tensor*>& GlMd, double thr, size_t batch_sz)
: dl_safezone_function(GlMd), threshold(thr), batch_size(batch_sz)
{ 
	num_of_params=0;
	for(auto layer:GlMd){
		num_of_params+=layer->size();
	}
	hyperparameters.push_back((float)thr);
	hyperparameters.push_back((float)batch_sz);
}

double Param_Variance_safezone_func::checkIfAdmissible(const size_t counter) const
{ 
	return (double)(batch_size-counter); 
}

/*
double Param_Variance_safezone_func::checkIfAdmissible(const resizable_tensor& mdl) const
{
	resizable_tensor subtr(GlobalModel.size(),1,1,1);
	for(size_t i=0;i<subtr.size();i++){
		subtr.host()[i]=mdl.host()[i]-GlobalModel.host()[i];
	}
	return threshold - dlib::dot(subtr, subtr);
}*/

double Param_Variance_safezone_func::checkIfAdmissible(const vector<tensor*>& mdl) const
{	
	double res=0;
	for(size_t i=0;i<mdl.size();i++){
		resizable_tensor subtr(*mdl.at(i));
		subtr*=-1.;
		dlib::cuda::add(1.,subtr,1.,*GlobalModel.at(i));
		res+=dot(subtr,subtr);
	}
	return threshold-res;
}

double Param_Variance_safezone_func::checkIfAdmissible(const vector<resizable_tensor*>& mdl) const
{	
	double res=0;
	for(size_t i=0;i<mdl.size();i++){
		resizable_tensor subtr(*mdl.at(i));
		subtr*=-1.;
		dlib::cuda::add(1.,subtr,1.,*GlobalModel.at(i));
		res+=dot(subtr,subtr);
	}
	return threshold-res;
}

Param_Variance_safezone_func::~Param_Variance_safezone_func() { }


/*********************************************
	dl_safezone
*********************************************/

dl_safezone::dl_safezone() 
: szone(nullptr) { }

dl_safezone::~dl_safezone() { }

// valid dl_safezone
dl_safezone::dl_safezone(dl_safezone_function* sz)
: szone(sz) { }

// Movable
dl_safezone::dl_safezone(dl_safezone&& other)
{
	swap(other);
}

dl_safezone& dl_safezone::operator=(dl_safezone&& other)
{
	swap(other);
	return *this;
}

// Copyable
dl_safezone::dl_safezone(const dl_safezone& other){
	szone = other.szone;
}

dl_safezone& dl_safezone::operator=(const dl_safezone& other){
	if(szone != other.szone) {
		szone = other.szone;
	}
	return *this;
}


/*********************************************
	dl_query_state
*********************************************/

dl_query_state::dl_query_state()
{ }

/*
dl_query_state::dl_query_state(size_t sz){ 
	GlobalModel.set_size(sz,1,1,1);
	GlobalModel=0.;
}
*/
dl_query_state::dl_query_state(vector<resizable_tensor*>& mdl){ 
	num_of_params=0;
	for(auto layer:mdl){
		resizable_tensor* l;
		l=new resizable_tensor();
		l->set_size(layer->num_samples(), layer->k(), layer->nr(), layer->nc());
		*l=0.;
		GlobalModel.push_back(l);
		num_of_params+=layer->size();
	}
}

dl_query_state::~dl_query_state() { }

dl_safezone_function* dl_query_state::dl_safezone(string cfg, string algo){
	
	Json::Value root;
	std::ifstream cfgfl(cfg);
	cfgfl >> root;
	
	string algorithm = root[algo].get("algorithm","Batch_Learning").asString();
	if(algorithm == "Batch_Learning"){
		auto safe_zone = new Batch_safezone_function(GlobalModel, root[algo].get("batch_size",32).asInt64());
		return safe_zone;
	}else if(algorithm == "Michael_Kamp"){
		auto safe_zone = new Param_Variance_safezone_func(GlobalModel,
		                                                  root[algo].get("threshold",1.).asDouble(),
													      root[algo].get("batch_size",32).asInt64());
		return safe_zone; 
	}else{
		return nullptr;
	}	
}


/*********************************************
	tcp_channel
*********************************************/

tcp_channel::tcp_channel(host* src, host* dst, rpcc_t endp)
    : channel(src, dst, endp), tcp_byts(0)
{ }

void tcp_channel::transmit(size_t msg_size)
{
    // update parent statistics
    channel::transmit(msg_size);

    // update tcp byte count
    size_t segno = (msg_size + tcp_mss - 1)/tcp_mss;
    tcp_byts += msg_size + segno * tcp_header_bytes;
}