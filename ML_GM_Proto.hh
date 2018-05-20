#ifndef __ML_GM_PROTO_HH__
#define __ML_GM_PROTO_HH__

#include <cmath>
#include <map>
#include <typeinfo>
#include <typeindex>
#include <stdexcept>
#include <cassert>
#include <ctime>
//#include <arma_extend.hpp>

#include "Machine_Learning.hh"
#include "dsource.hh"
#include "dsarch.hh"
#include <dlib/cuda/tensor_tools.h>

/**
	\file Distributed stream system architecture simulation classes for distributed machine learning.
  */

namespace ml_gm_proto{

using namespace dds;
using namespace machine_learning;
using namespace machine_learning::MLPACK_Classification;
using namespace machine_learning::MLPACK_Regression;
using namespace machine_learning::DLIB_Classification;
using namespace machine_learning::DLIB_Regression;
using std::map;
using std::string;
using std::vector;
using std::cout;
using std::endl;
	
/**
	A channel implementation which accounts a combined network cost.

	The network cost is computed as follows: 
	for each transmit of b bytes, there is a total charge of
	header * ceiling(b/MSS) bytes.

	This cost is resembling the TPC segment cost.
  */
struct tcp_channel : channel
{
	static constexpr size_t tcp_header_bytes = 40;
	static constexpr size_t tcp_mss = 1024;

	tcp_channel(host* src, host* dst, rpcc_t endp);

	void transmit(size_t msg_size) override;

	inline size_t tcp_bytes() const { return tcp_byts; }

protected:
	size_t tcp_byts;

};
	
namespace MlPack_GM_Proto{

/**
	Wrapper for a state parameters.
	
	This class wraps a reference to the parameters of a model
	together with a count of the updates it contains since the
	last synchronization.
  */
struct model_state
{
	const arma::mat& _model;
	size_t updates;

	inline model_state(const arma::mat& _mdl, size_t _updates)
		: _model(_mdl), updates(_updates) { }

	size_t byte_size() const { return (_model.n_elem)*sizeof(double); }
};


/** 
	The base class of a safezone function for machine learning purposes.
	*/
struct ml_safezone_function {
	
	arma::mat& GlobalModel; // The global model.
	
	ml_safezone_function(arma::mat& mdl);
	virtual ~ml_safezone_function();
	
	const arma::mat& getGlobalModel() const;
	virtual double checkIfAdmissible(const size_t counter) const { return 0.; }
	virtual double checkIfAdmissible(const arma::mat&) const { return 0.; }
	virtual double checkIfAdmissible(const arma::cube&) const { return 0.; } 
	virtual size_t byte_size() const { return 0; }
	virtual void pr() { cout << endl << "Simple safezone function." << endl; }
};


/**
	This safezone function just checks the number of points
	a local site has proccesed since the last synchronisation. The threshold
	variable basically indicates the batch size. If the proccesed points reach
	the batch size, then the function returns an inadmissible region.
 */
struct Batch_Learning : ml_safezone_function {
	
	size_t threshold; // The maximum number of points fitted by each node before requesting synch from the Hub.
	
	Batch_Learning(arma::mat& GlMd);
	Batch_Learning(arma::mat& GlMd, size_t thr);
	~Batch_Learning();
	
	double checkIfAdmissible(const size_t counter) const override;
	size_t byte_size() const override { return GlobalModel.n_elem*sizeof(double) + sizeof(size_t); }
};

/**
	This safezone function implements the algorithm presented in
	in the paper "Communication-Efficient Distributed Online Prediction
	by Dynamic Model Synchronization"
	by Michael Kamp, Mario Boley, Assaf Schuster and Izchak Sharfman.
 */
struct Variance_safezone_func : ml_safezone_function {
	
	double threshold; // The threshold of the variance between the models of the network.
	size_t batch_size; // The number of points seen by the node since the last synchronization.
	
	Variance_safezone_func(arma::mat& GlMd);
	Variance_safezone_func(arma::mat& GlMd, size_t batch_sz);
	Variance_safezone_func(arma::mat& GlMd, double thr);
	Variance_safezone_func(arma::mat& GlMd, double thr, size_t batch_sz);
	~Variance_safezone_func();
	
	double checkIfAdmissible(const size_t counter) const override;
	double checkIfAdmissible(const arma::mat& mdl) const override;
	size_t byte_size() const override { return (1+GlobalModel.n_elem)*sizeof(double) + sizeof(size_t); }
};

/** 
	A wrapper containing the safezone function for machine
	learning purposes.
	
	It is essentially a wrapper for the more verbose, polymorphic \c safezone_func API,
	but it conforms to the standard functional API. It is copyable and in addition, it
	provides a byte_size() method, making it suitable for integration with the middleware.
	*/
class safezone {
	ml_safezone_function* szone;        // the safezone function, if any
public:

	/// null state
	safezone();
	~safezone();
	
	/// valid safezone
	safezone(ml_safezone_function* sz);
	//~safezone();
	
	/// Movable
	safezone(safezone&&);
	safezone& operator=(safezone&&);
	
	/// Copyable
	safezone(const safezone&);
	safezone& operator=(const safezone&);
	
	inline void swap(safezone& other) {
		std::swap(szone, other.szone);
	}
	
	ml_safezone_function* getSZone() { return (szone!=nullptr) ? szone : nullptr; }
	
	inline double operator()(const size_t counter)
	{
		return (szone!=nullptr) ? szone->checkIfAdmissible(counter) : NAN;
	}
	
	inline double operator()(const arma::mat& mdl)
	{
		return (szone!=nullptr) ? szone->checkIfAdmissible(mdl) : NAN;
	}
	
	inline double operator()(const arma::cube& mdls)
	{
		return (szone!=nullptr) ? szone->checkIfAdmissible(mdls) : NAN;
	}
	
	inline size_t byte_size() const {
		return (szone!=nullptr) ? szone->byte_size() : 0;
	}
	
};


/**
	Base class for a query state object.

	A query state holds the current global estimate model. It also honls the
	accuracy of the current global model (percentage of correctly classified
	datapoins in case of classification and RMSE score in case of regression).
 */
struct query_state
{
	arma::mat GlobalModel;  // The global model.
	double accuracy; // The accuracy of the current global model.
	
	query_state();
	query_state(arma::SizeMat sz);
	virtual ~query_state();
	
	void initializeGlobalModel(arma::SizeMat sz) { GlobalModel = arma::mat(sz, arma::fill::zeros); }
	
	/** Update the global model parameters.
		
		After this function, the query estimate, accuracy and
		safezone should adjust to the new global model. For now
		only the global model is adjusted.
		*/
	void update_estimate(arma::mat& mdl) { GlobalModel -= GlobalModel - mdl; }
	
	/**
		Return a ml_safezone_func for the safe zone function.

	 	The returned object shares state with this object.
	 	It is the caller's responsibility to delete the returned object,
		and do so before this object is destroyed.
	 */
	ml_safezone_function* safezone(string cfg, string algo);
	
	virtual size_t byte_size() const { return (1+GlobalModel.n_rows)*sizeof(double); }
	
};


/**
	Query and protocol configuration.
  */
struct protocol_config
{
	string learning_algorithm;              // options : [ PA, KernelPA, MLP, PA_Reg, NN_Reg]
	string distributed_learning_algorithm;  // options : [ Batch_Learning, Michael_Kmp, Michael_Kmp_Kernel ]
	string cfgfile;                         // The JSON file containing the info for the test.
	string network_name;                    // The name of the network being queried.
};


/** 
	A base class for a continuous query.
	Objects inheriting this class must override the virtual methods.
	*/
struct continuous_query 
{
	// These are attributes requested by the user
	protocol_config config;
	
	arma::mat* testSet;         // Test dataset without labels.
	arma::mat* testResponses;   // Labels of the test dataset.
	
	continuous_query(arma::mat* tSet, arma::mat* tRes, string cfg, string nm);
	virtual ~continuous_query() { }
	
	inline query_state* create_query_state() { return new query_state(); }
	inline query_state* create_query_state(arma::SizeMat sz) { return new query_state(sz); }
	virtual inline double queryAccuracy(MLPACK_Learner* lnr) { return 0.; }
};

struct Classification_query : continuous_query
{
	/** Constructor */
	Classification_query(arma::mat* tSet, arma::mat* tRes, string cfg, string nm);
	/** Destructor */
	~Classification_query() { delete testSet; delete testResponses; }
	
	double queryAccuracy(MLPACK_Learner* lnr) override;
	
};

struct Regression_query : continuous_query
{
	/** Constructor */
	Regression_query(arma::mat* tSet, arma::mat* tRes, string cfg, string nm);
	/** Destructor */
	~Regression_query() { delete testSet; delete testResponses; }
	
	double queryAccuracy(MLPACK_Learner* lnr) override;
};

/**
	The star network topology using the Geometric Method 
	for Distributed Machine Learning.
	
	*/
template <typename Net, typename Coord, typename Node>
struct gm_learning_network : star_network<Net, Coord, Node>
{
	typedef Coord coordinator_t;
	typedef Node node_t;
	typedef Net network_t;
	typedef star_network<network_t, coordinator_t, node_t> star_network_t;

	continuous_query* Q;
	
	const protocol_config& cfg() const { return Q->config; }

	gm_learning_network(const set<source_id>& _hids, const string& _name, continuous_query* _Q)
	: star_network_t(_hids), Q(_Q) 
	{ 
		this->set_name(_name);
		this->setup(Q);
	}

	channel* create_channel(host* src, host* dst, rpcc_t endp) const override
	{
		if(! dst->is_mcast())
			return new tcp_channel(src, dst, endp);
		else
			return basic_network::create_channel(src, dst, endp);
	}
	
	// This is called to update a specific learning node in the network.
	void process_record(size_t randSite, arma::mat& batch, arma::mat& labels){
		this->source_site( this->sites.at(randSite)->site_id() )->update_stream(batch, labels);		
	}

	virtual void warmup(arma::mat& batch, arma::mat& labels){
		// let the coordinator initialize the nodes
		this->hub->warmup(batch,labels);
	}
	
	virtual void start_round(){
		this->hub->start_round();
	}

	virtual void process_fini(){
		this->hub->finish_rounds();
	}

	~gm_learning_network() { delete Q; }
};

} //*  End namespace MlPack_GM_Proto *//

namespace DLib_GM_Proto{

	
/**
	Wrapper for a state parameters.
	
	This class wraps a reference to the parameters of a model
	together with a count of the updates it contains since the
	last synchronization.
  */
struct dl_model_state
{
	const vector<tensor*>& _model;
	size_t updates;
	size_t num_of_params;

	inline dl_model_state(const vector<tensor*>& _mdl, size_t _updates)
	: _model(_mdl), updates(_updates) {
		num_of_params=0;
		for(auto layer:_mdl){
			num_of_params+=layer->size();
		}
	}

	size_t byte_size() const { return num_of_params*sizeof(float); }
};


struct tensor_message
{
	const vector<resizable_tensor*>& _model;
	size_t updates;
	size_t num_of_params;

	inline tensor_message(const vector<resizable_tensor*>& _mdl, size_t _updates)
	: _model(_mdl), updates(_updates) {
		num_of_params=0;
		for(auto layer:_mdl){
			num_of_params+=layer->size();
		}
	}

	size_t byte_size() const { return num_of_params*sizeof(float); }
};


/** 
	The base class of a safezone function for machine learning purposes.
	*/
struct dl_safezone_function {
	
	//resizable_tensor& GlobalModel; // The global model.
	vector<resizable_tensor*>& GlobalModel; // The global model.
	size_t num_of_params; // The number on parameters.
	vector<float> hyperparameters; // Avector of hyperparameters.
	
	//dl_safezone_function(resizable_tensor& mdl);
	dl_safezone_function(vector<resizable_tensor*>& mdl);
	virtual ~dl_safezone_function();
	
	//const resizable_tensor& getGlobalModel() const;
	const vector<resizable_tensor*>& getGlobalModel() const;
	virtual double checkIfAdmissible(const size_t counter) const { return 0.; }
	//virtual double checkIfAdmissible(const resizable_tensor& pars) const { return 0.; }
	virtual double checkIfAdmissible(const vector<resizable_tensor*>& pars) const { return 0.; }
	virtual double checkIfAdmissible(const vector<tensor*>& pars) const { return 0.; }
	virtual size_t byte_size() const { return 0; }
	vector<float> hyper() const { return hyperparameters; }
	virtual void pr() { cout << endl << "Simple safezone function." << endl; }
};


/**
	This safezone function just checks the number of points
	a local site has proccesed since the last synchronisation. The threshold
	variable basically indicates the batch size. If the proccesed points reach
	the batch size, then the function returns an inadmissible region.
 */
struct Batch_safezone_function : dl_safezone_function {
	
	size_t threshold; // The maximum number of points fitted by each node before requesting synch from the Hub.
	
	//Batch_safezone_function(resizable_tensor& GlMd);
	//Batch_safezone_function(resizable_tensor& GlMd, size_t thr);
	Batch_safezone_function(vector<resizable_tensor*>& GlMd);
	Batch_safezone_function(vector<resizable_tensor*>& GlMd, size_t thr);
	~Batch_safezone_function();
	
	double checkIfAdmissible(const size_t counter) const override;
	//size_t byte_size() const override { return GlobalModel.size()*sizeof(float) + sizeof(size_t); }
	size_t byte_size() const override { return num_of_params*sizeof(float) + sizeof(size_t); }
};

/**
	This safezone function implements the algorithm presented in
	in the paper "Communication-Efficient Distributed Online Prediction
	by Dynamic Model Synchronization"
	by Michael Kamp, Mario Boley, Assaf Schuster and Izchak Sharfman.
 */
struct Param_Variance_safezone_func : dl_safezone_function {
	
	double threshold; // The threshold of the variance between the models of the network.
	size_t batch_size; // The number of points seen by the node since the last synchronization.
	
	//Param_Variance_safezone_func(resizable_tensor& GlMd);
	//Param_Variance_safezone_func(resizable_tensor& GlMd, size_t batch_sz);
	//Param_Variance_safezone_func(resizable_tensor& GlMd, double thr);
	//Param_Variance_safezone_func(resizable_tensor& GlMd, double thr, size_t batch_sz);
	
	Param_Variance_safezone_func(vector<resizable_tensor*>& GlMd);
	Param_Variance_safezone_func(vector<resizable_tensor*>& GlMd, size_t batch_sz);
	Param_Variance_safezone_func(vector<resizable_tensor*>& GlMd, double thr);
	Param_Variance_safezone_func(vector<resizable_tensor*>& GlMd, double thr, size_t batch_sz);
	
	~Param_Variance_safezone_func();
	
	double checkIfAdmissible(const size_t counter) const override;
	//double checkIfAdmissible(const resizable_tensor& pars) const override;
	double checkIfAdmissible(const vector<resizable_tensor*>& pars) const override;
	double checkIfAdmissible(const vector<tensor*>& mdl) const override;
	//size_t byte_size() const override { return GlobalModel.size()*sizeof(float) + sizeof(double) + sizeof(size_t); }
	size_t byte_size() const override { return num_of_params*sizeof(float) + sizeof(double) + sizeof(size_t); }
};

/** 
	A wrapper containing the safezone function for machine
	learning purposes.
	
	It is essentially a wrapper for the more verbose, polymorphic \c safezone_func API,
	but it conforms to the standard functional API. It is copyable and in addition, it
	provides a byte_size() method, making it suitable for integration with the middleware.
	*/
class dl_safezone {
	dl_safezone_function* szone;        // the safezone function, if any
public:

	/// null state
	dl_safezone();
	~dl_safezone();
	
	/// valid safezone
	dl_safezone(dl_safezone_function* sz);
	//~safezone();
	
	/// Movable
	dl_safezone(dl_safezone&&);
	dl_safezone& operator=(dl_safezone&&);
	
	/// Copyable
	dl_safezone(const dl_safezone&);
	dl_safezone& operator=(const dl_safezone&);
	
	inline void swap(dl_safezone& other) {
		std::swap(szone, other.szone);
	}
	
	dl_safezone_function* getSZone() { return (szone!=nullptr) ? szone : nullptr; }
	
	inline double operator()(const size_t counter)
	{
		return (szone!=nullptr) ? szone->checkIfAdmissible(counter) : NAN;
	}
	
	//inline double operator()(const resizable_tensor& mdl)
	//{
	//	return (szone!=nullptr) ? szone->checkIfAdmissible(mdl) : NAN;
	//}
	
	inline double operator()(const vector<tensor*>& mdl)
	{
		return (szone!=nullptr) ? szone->checkIfAdmissible(mdl) : NAN;
	}
	
	inline double operator()(const vector<resizable_tensor*>& mdl)
	{
		return (szone!=nullptr) ? szone->checkIfAdmissible(mdl) : NAN;
	}
	
	inline size_t byte_size() const {
		return (szone!=nullptr) ? szone->byte_size() : 0;
	}
	
};


/**
	Base class for a query state object.

	A query state holds the current global estimate model. It also honls the
	accuracy of the current global model (percentage of correctly classified
	datapoins in case of classification and RMSE score in case of regression).
 */
struct dl_query_state
{
	//resizable_tensor GlobalModel;  // The global model.
	vector<resizable_tensor*> GlobalModel;  // The global model.
	double accuracy; // The accuracy of the current global model.
	size_t num_of_params; // The number on parameters.
	
	dl_query_state();
	//dl_query_state(size_t sz);
	dl_query_state(vector<resizable_tensor*>& mdl);
	virtual ~dl_query_state();
	
	//void initializeGlobalModel(size_t sz) { GlobalModel.set_size(sz,1,1,1); GlobalModel=0; }
	void initializeGlobalModel(const vector<tensor*>& mdl) {
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
	
	/** Update the global model parameters.
		
		After this function, the query estimate, accuracy and
		safezone should adjust to the new global model. For now
		only the global model is adjusted.
		*/
	//void update_estimate(resizable_tensor& mdl) { GlobalModel = resizable_tensor(mdl); }
	void update_estimate(vector<resizable_tensor*>& mdl) {
		for(size_t i=0;i<GlobalModel.size();i++){
			dlib:memcpy(*GlobalModel.at(i),*mdl.at(i));
		}
	}
	
	/**
		Return a ml_safezone_func for the safe zone function.

	 	The returned object shares state with this object.
	 	It is the caller's responsibility to delete the returned object,
		and do so before this object is destroyed.
	    */
	dl_safezone_function* dl_safezone(string cfg, string algo);
	
	//virtual size_t byte_size() const { return GlobalModel.size()*sizeof(float)+sizeof(double); }
	virtual size_t byte_size() const { return num_of_params*sizeof(float)+sizeof(double); }
	
};


/**
	Query and protocol configuration.
  */
struct dl_protocol_config
{
	string learning_algorithm;              // options : [ PA, KernelPA, MLP, PA_Reg, NN_Reg]
	string distributed_learning_algorithm;  // options : [ Batch_Learning, Michael_Kmp, Michael_Kmp_Kernel ]
	string cfgfile;                         // The JSON file containing the info for the test.
	string network_name;                    // The name of the network being queried.
	int image_width = 28;                   // The pixel width of the image.
	int image_height = 28;                  // The pixel height of the image.
	int number_of_channels = 1;             // The number of channels of the image. (i.e. 3 in case of RGB image)
};


/** 
	A base class for a continuous query.
	Objects inheriting this class must override the virtual methods.
	*/
template<typename feat, typename label>
struct dl_continuous_query 
{
	// These are attributes requested by the user
	dl_protocol_config config;
	
	std::vector<matrix<feat>>* testSet;         // Test dataset without labels.
	std::vector<label>* testResponses;   // Labels of the test dataset.
	
	dl_continuous_query(std::vector<matrix<feat>>* tSet, std::vector<label>* tRes, string cfg, string nm);
	virtual ~dl_continuous_query() { }
	
	inline dl_query_state* create_query_state() { return new dl_query_state(); }
	inline dl_query_state* create_query_state(vector<resizable_tensor*>& mdl) { return new dl_query_state(mdl); }
	virtual inline double queryAccuracy(DLIB_Learner<feat,label>* lnr) { return 0.; }
};

template<typename feat, typename label>
dl_continuous_query<feat,label>::dl_continuous_query(std::vector<matrix<feat>>* tSet, std::vector<label>* tRes, string cfg, string nm)
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
	config.image_width = root["gm_network_"+nm]
						 .get("image_width", 0).asInt64();
	config.image_height = root["gm_network_"+nm]
						  .get("image_height", 0).asInt64();
	config.number_of_channels = root["gm_network_"+nm]
						        .get("number_of_channels", 0).asInt64();
	config.cfgfile = cfg;
	
	cout << "Query initialized : " << config.learning_algorithm << ", ";
	cout << config.distributed_learning_algorithm << ", ";
	cout << config.network_name << ", ";
	cout << config.cfgfile << endl;
}


template<typename feat, typename label>
struct dl_Classification_query : dl_continuous_query<feat,label>
{
	/** Constructor */
	dl_Classification_query(std::vector<matrix<feat>>* tSet, std::vector<label>* tRes, string cfg, string nm);
	/** Destructor */
	~dl_Classification_query() { delete this->testSet; delete this->testResponses; }
	
	double queryAccuracy(DLIB_Learner<feat,label>* lnr) override;
	
};

template<typename feat, typename label>
dl_Classification_query<feat,label>::dl_Classification_query(std::vector<matrix<feat>>* tSet, std::vector<label>* tRes, string cfg, string nm)
:dl_continuous_query<feat,label>(tSet,tRes,cfg,nm) { }

template<typename feat, typename label>
double dl_Classification_query<feat,label>::queryAccuracy(DLIB_Learner<feat,label>* lnr) {	
	return lnr->accuracy(*this->testSet,*this->testResponses);	
}


template<typename feat, typename label>
struct dl_Regression_query : dl_continuous_query<feat,label>
{
	/** Constructor */
	dl_Regression_query(std::vector<matrix<feat>>* tSet, std::vector<label>* tRes, string cfg, string nm);
	/** Destructor */
	~dl_Regression_query() { delete this->testSet; delete this->testResponses; }
	
	double queryAccuracy(DLIB_Learner<feat,label>* lnr) override;
};

template<typename feat, typename label>
dl_Regression_query<feat,label>::dl_Regression_query(std::vector<matrix<feat>>* tSet, std::vector<label>* tRes, string cfg, string nm)
:dl_continuous_query<feat,label>(tSet,tRes,cfg,nm) { }

template<typename feat, typename label>
double dl_Regression_query<feat,label>::queryAccuracy(DLIB_Learner<feat,label>* lnr) {
	return lnr->accuracy(*this->testSet,*this->testResponses);
}

/**
	The star network topology using the Geometric Method 
	for Distributed Machine Learning.
	*/
template <typename feat, typename lb, template<typename,typename> typename Net, template<typename,typename> typename Coord, template<typename,typename> typename Node>
struct dl_gm_learning_network : star_network<Net<feat,lb>, Coord<feat,lb>, Node<feat,lb>>
{
	typedef Coord<feat,lb> coordinator_t;
	typedef Node<feat,lb> node_t;
	typedef Net<feat,lb> network_t;
	typedef feat features_t;
	typedef lb labels_t;
	typedef star_network<network_t, coordinator_t, node_t> star_network_t;

	dl_continuous_query<feat,lb>* Q;
	
	const dl_protocol_config& cfg() const { return Q->config; }

	dl_gm_learning_network(const set<source_id>& _hids, const string& _name, dl_continuous_query<feat,lb>* _Q)
	: star_network_t(_hids), Q(_Q) 
	{ 
		this->set_name(_name);
		this->setup(Q);
	}

	channel* create_channel(host* src, host* dst, rpcc_t endp) const override
	{
		if(! dst->is_mcast())
			return new tcp_channel(src, dst, endp);
		else
			return basic_network::create_channel(src, dst, endp);
	}
	
	/*
	/// This is called to update a specific learning node in the network.
	void warmup(size_t site, std::vector<matrix<feat>>& batch, std::vector<lb>& labels){
		this->source_site( this->sites.at(site)->site_id() )->warmup(batch, labels);		
	}
	
	/// This is called to update a specific learning node in the network.
	void end_warmup(size_t site){
		this->source_site( this->sites.at(site)->site_id() )->end_warmup();		
	}*/
	
	virtual void warmup(std::vector<matrix<feat>>& batch, std::vector<lb>& labels){
		// let the coordinator initialize the nodes
		this->hub->warmup(batch,labels);
	}
	
	/// This is called to update a specific learning node in the network.
	void end_warmup(){
		this->hub->end_warmup();		
	}
	
	virtual void start_round(){
		this->hub->start_round();
	}
	
	/// This is called to update a specific learning node in the network.
	void process_record(size_t site, std::vector<matrix<feat>>& batch, std::vector<lb>& labels){
		this->source_site( this->sites.at(site)->site_id() )->update_stream(batch, labels);		
	}

	virtual void process_fini(){
		this->hub->finish_rounds();
	}

	~dl_gm_learning_network() { delete Q; }
};
	
} //*  End namespace DLib_GM_Proto *//

} // end namespace ml_gm_proto

#endif