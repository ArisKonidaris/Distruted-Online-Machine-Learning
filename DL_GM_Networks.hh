#ifndef __DL_GM_NETWORKS_HH__
#define __DL_GM_NETWORKS_HH__

#include <boost/range/adaptors.hpp>
#include <boost/shared_ptr.hpp>
#include <random>

#include "ML_GM_Proto.hh"

namespace ml_gm_proto {
	
using namespace dds;
using namespace H5;
using namespace data_src;
using std::map;
using std::cout;
using std::endl;

namespace DL_gm_networks {

using namespace ml_gm_proto::DLib_GM_Proto;

//struct node_proxy;
template<typename feat, typename lb>
struct coordinator;
template<typename feat, typename lb>
struct learning_node;
template<typename feat, typename lb>
struct learning_node_proxy;

template<typename feat,typename lb>
struct network : dl_gm_learning_network<feat, lb, network, coordinator, learning_node>{	
	network(const set<source_id>& _hids, const string& _name, dl_continuous_query<feat,lb>* _Q);
};

template<typename feat,typename lb>
network<feat,lb>::network(const set<source_id>& _hids, const string& _name, dl_continuous_query<feat,lb>* _Q)
: dl_gm_learning_network<feat, lb, network, coordinator, learning_node>(_hids, _name, _Q){
	this->set_protocol_name("LD_GM");
}

template<typename feat,typename lb>
struct coordinator : process
{
	typedef feat feature_t;
	typedef lb label_t;
	typedef coordinator<feature_t,label_t> coordinator_t;
	typedef learning_node<feature_t,label_t> node_t;
	typedef learning_node_proxy<feature_t,label_t> node_proxy_t;
	typedef network<feature_t,label_t> network_t;

	proxy_map<node_proxy_t, node_t> proxy;

	//
	// protocol stuff
	//
	DLIB_Learner<feat,lb>* global_learner;
    dl_continuous_query<feature_t,label_t>* Q;   		              // continuous query
    dl_query_state* query; 		              // current query state
 	dl_safezone_function* safe_zone; 	      // the safe zone wrapper

	size_t k;					              // number of sites

	// index the nodes
	map<node_t*, size_t> node_index;
	vector<node_t*> node_ptr;

	coordinator(network_t* nw, dl_continuous_query<feature_t, label_t>* _Q); 
	~coordinator();

	inline network_t* net() { return static_cast<network_t*>(host::net()); }
	inline const dl_protocol_config& cfg() const { return Q->config; }

	// Initialize the Learner and its' variables.
	void initializeLearner();

	void setup_connections() override;

	// initialize a new round
	void start_round();
	void finish_round();
	void finish_rounds();
	
	// rebalance algorithm by Kamp
	void Kamp_Rebalance(node_t* n);

	// get the model of a node
	void fetch_updates(node_t* n);
	
	void Progress();

	resizable_tensor computeGlobalModel(const resizable_tensor& Param);

	// remote call on warmed up host
	oneway end_warmup(sender<node_t> ctx);

	// remote call on host violation
	oneway local_violation(sender<node_t> ctx);
	
	set<node_t*> B;					 // initialized by local_violation(), 
								     // updated by rebalancing algo

	set<node_t*> Bcompl;		     // complement of B, updated by rebalancing algo
	
	vector<resizable_tensor*> Mean;   // Used to compute the mean model
	size_t num_violations;           // Number of violations in the same round (for rebalancing)

	// statistics
	size_t num_rounds;				 // total number of rounds
	size_t num_subrounds;			 // total number of subrounds
	size_t sz_sent;					 // total safe zones sent
 	size_t total_updates;		     // number of stream updates received

};

template<typename feat,typename lb>
void coordinator<feat,lb>::start_round(){
	for(auto n : net()->sites) {
		sz_sent ++;
		proxy[n].reset(dl_safezone(safe_zone));
	}
	num_rounds++;
}

template<typename feat,typename lb>
oneway coordinator<feat,lb>::end_warmup(sender<node_t> ctx){
	node_t* n = ctx.value;
	fetch_updates(n);
	B.insert(n);
	if(B.size()==k){
		for(auto layer:Mean){
			dlib::cuda::affine_transform(*layer,*layer,(float)std::pow(k,-1));
		}
		//Mean/=(float)k;
		B.clear();
		query->update_estimate(Mean);
		global_learner->update_model(Mean);
		start_round();
	}
}

template<typename feat,typename lb>
oneway coordinator<feat,lb>::local_violation(sender<node_t> ctx){
	
	node_t* n = ctx.value;
	num_violations++;
	
	B.clear(); // Clear the balanced nodes set.
	for(auto layer:Mean){
		dlib::cuda::affine_transform(*layer,*layer,0.);
	}
	//Mean=0.; // Clear the mean global model.
	
	if( dl_safezone_function* entity = dynamic_cast<Batch_safezone_function*>(safe_zone) ){
		num_violations = 0;
		finish_round();
	}else{
		if(num_violations==k){
			num_violations = 0;
			finish_round();
		}else{
			Kamp_Rebalance(n);
		}
	}
}

/*
template<typename feat,typename lb>
void coordinator<feat,lb>::fetch_updates(node_t* node){
	dl_model_state up = proxy[node].get_drift();
	if(query->GlobalModel.size()==0){
		size_t sz=0;
		for(auto layer:up._model){
			sz+=layer->size();
		}
		query->initializeGlobalModel(sz);
		Mean.set_size(sz,1,1,1);
		Mean=0.;
	}
	size_t sz=0;
	for(auto layer:up._model){
		for(size_t i=0;i<layer->size();i++){
			Mean.host()[sz]+=layer->host()[i];
			sz++;
		}
	}
	total_updates += up.updates;
}
*/

template<typename feat,typename lb>
void coordinator<feat,lb>::fetch_updates(node_t* node){
	dl_model_state up = proxy[node].get_drift();
	if(query->GlobalModel.size()==0){
		query->initializeGlobalModel(up._model);
		for(auto layer:up._model){
			resizable_tensor* l;
			l=new resizable_tensor();
			l->set_size(layer->num_samples(),layer->k(),layer->nr(),layer->nc());
			*l=0.;
			Mean.push_back(l);
		}
	}
	for(size_t i=0;i<up._model.size();i++){
		dlib::cuda::add(1.,*Mean.at(i),1.,*up._model.at(i));
	}
	total_updates += up.updates;
}

// initialize a new round
template<typename feat,typename lb>
void coordinator<feat,lb>::finish_round() {

	// Collect all data
	for(auto n : node_ptr) {
		fetch_updates(n);
	}
	for(auto layer:Mean){
		dlib::cuda::affine_transform(*layer,*layer,(float)std::pow(k,-1));
	}
	//Mean /= (float)k;

	// New round
	query->update_estimate(Mean);
	global_learner->update_model(Mean);
	
	start_round();

}

template<typename feat,typename lb>
void coordinator<feat,lb>::Kamp_Rebalance(node_t* lvnode){
	
	Bcompl.clear();
	B.insert(lvnode);
	
	// find a balancing set
	vector<node_t*> nodes;
	nodes.reserve(k);
	for(auto n:node_ptr){
		if(B.find(n) == B.end())
			nodes.push_back(n);
	}
	assert(nodes.size()==k-1);
	
	// permute the order
	std::random_shuffle(nodes.begin(), nodes.end());
	assert(nodes.size()==k-1);
	assert(B.size()==1);
	assert(Bcompl.empty());
	
	for(auto n:nodes){
		Bcompl.insert(n);
	}
	assert(B.size()+Bcompl.size()==k);
	
	fetch_updates(lvnode);
	for(auto n:Bcompl){
		fetch_updates(n);
		B.insert(n);
		for(auto layer:Mean){
			dlib::cuda::affine_transform(*layer,*layer,(float)std::pow(B.size(),-1));
		}
		//Mean /=(float)B.size();
		if(safe_zone->checkIfAdmissible(Mean) > 0. || B.size() == k )
			break;
		for(auto layer:Mean){
			dlib::cuda::affine_transform(*layer,*layer,(float)B.size());
		}
		//Mean *=(float)B.size();
	}
	
	if(B.size() < k){ 
		// Rebalancing
		for(auto n : B) {
			proxy[n].set_drift(tensor_message(Mean, 0));
			//proxy[n].set_drift(dl_model_state(Mean, 0));
		}
		num_subrounds++;
	}else{
		// New round
		num_violations = 0;
		query->update_estimate(Mean);
		global_learner->update_model(Mean);
		start_round();
	}
	
}

template<typename feat,typename lb>
void coordinator<feat,lb>::Progress(){
	
	cout << "Global model of network " << net()->name() << "." << endl;

	// Query thr accuracy of the global model.
	if(Q->config.learning_algorithm == "LeNet"){
		query->accuracy = Q->queryAccuracy(global_learner);
	}
	cout << "accuracy : " << std::setprecision(6) << (float)100.*query->accuracy << "%" << endl;
	cout << "Number of rounds : " << num_rounds << endl;
	cout << "Number of subrounds : " << num_subrounds << endl;
	cout << "Total updates : " << total_updates << endl;
}

template<typename feat,typename lb>
void coordinator<feat,lb>::finish_rounds(){
	
	cout << endl;
	cout << "Global model of network " << net()->name() << "." << endl;
	cout << "tests : " << Q->testSet->size() << endl;
	
	// Query thr accuracy of the global model.
	query->accuracy = Q->queryAccuracy(global_learner);
	
	// See the total number of points received by all the nodes. For debugging.
	for(auto nd:node_ptr){
		total_updates += nd->_learner->getNumOfUpdates();
	}
	
	// Print the results.
	if(Q->config.learning_algorithm == "LeNet"){
		cout << "accuracy : " << std::setprecision(6) << (float)100.*query->accuracy << "%" << endl;
	}else{
		cout << "Another measure of accuracy for later on." << endl;
	}
	cout << "Number of rounds : " << num_rounds << endl;
	cout << "Number of subrounds : " << num_subrounds << endl;
	cout << "Total updates : " << total_updates << endl;
	
}

template<typename feat,typename lb>
void coordinator<feat,lb>::setup_connections(){
	using boost::adaptors::map_values;
	proxy.add_sites(net()->sites);
	for(auto n : net()->sites) {
		node_index[n] = node_ptr.size();
		node_ptr.push_back(n);
	}
	k = node_ptr.size();
}

template<typename feat,typename lb>
void coordinator<feat,lb>::initializeLearner(){
	if (cfg().learning_algorithm == "LeNet"){
		global_learner = new LeNet<feature_t,label_t>();
		
		std::vector<matrix<feature_t>> random_image_init;
		std::vector<label_t> random_label_init;
		matrix<feature_t, 28, 28> random_image;
		for(size_t rows=0;rows<28;rows++){
			for(size_t cols=0;cols<28;cols++){
				random_image(rows,cols)=(feature_t)std::rand()%(256);
			}
		}
		random_image_init.push_back(random_image);
		random_label_init.push_back((label_t)1);
		global_learner->fit(random_image_init,random_label_init);
	}
}

template<typename feat,typename lb>
coordinator<feat,lb>::coordinator(network_t* nw, dl_continuous_query<feat,lb>* _Q)
: 	process(nw), proxy(this),
	Q(_Q),
	k(0),
	num_violations(0), num_rounds(0), num_subrounds(0),
	sz_sent(0), total_updates(0){
	initializeLearner();
	query = Q->create_query_state();
	safe_zone = query->dl_safezone(cfg().cfgfile, cfg().distributed_learning_algorithm);
}

template<typename feat,typename lb>
coordinator<feat,lb>::~coordinator(){
	delete safe_zone;
	delete query;
}

template<typename feat,typename lb>
struct coord_proxy : remote_proxy<coordinator<feat,lb>>
{
	using coordinator_t = coordinator<feat,lb>;
	REMOTE_METHOD(coordinator_t, local_violation);
	REMOTE_METHOD(coordinator_t, end_warmup);
	coord_proxy(process* c) : remote_proxy<coordinator_t>(c) { } 
};


/**
	This is a site implementation for the classic Geometric Method protocol.

 */
template<typename feat,typename lb>
struct learning_node : local_site {

	typedef feat feature_t;
	typedef lb label_t;
	typedef coordinator<feature_t, label_t> coordinator_t;
	typedef learning_node<feature_t, label_t> node_t;
	typedef learning_node_proxy<feature_t, label_t> node_proxy_t;
	typedef network<feature_t, label_t> network_t;
	typedef coord_proxy<feature_t, label_t> coord_proxy_t;
    typedef dl_continuous_query<feature_t, label_t> continuous_query_t;

    continuous_query_t* Q;                  // The query management object.
    dl_safezone szone;                       // The safezone object.
	DLIB_Learner<feat,lb>* _learner;

	int num_sites;			             // Number of sites.
	size_t datapoints_seen;              // Number of points the node has seen since the last synchronization.
	coord_proxy_t coord;                 // The proxy of the coordinator/hub.

	learning_node(network_t* net, source_id hid, continuous_query_t* _Q)
	:	local_site(net, hid), Q(_Q), coord( this )
	{ 
		coord <<= net->hub;
		initializeLearner();
	};

	inline const dl_protocol_config& cfg() const { return Q->config; }

	void initializeLearner();

	void setup_connections() override;

	void warmup(std::vector<matrix<feature_t>>& batch, std::vector<label_t>& labels);
	
	void end_warmup();

	void update_stream(std::vector<matrix<feature_t>>& batch, std::vector<label_t>& labels);

	//
	// Remote methods
	//

	// called at the start of a round
	oneway reset(const dl_safezone& newsz); 

	// transfer data to the coordinator
	dl_model_state get_drift();
	
	// set the drift vector (for rebalancing)
	void set_drift(tensor_message mdl);
	//void set_drift(dl_model_state mdl);
	
};

template<typename feat,typename lb>
oneway learning_node<feat,lb>::reset(const dl_safezone& newsz){
	szone = newsz;       // Reset the safezone object
	datapoints_seen = 0; // Reset the drift vector
	_learner->update_model(szone.getSZone()->getGlobalModel()); // Updates the parameters of the local learner
}

template<typename feat,typename lb>
dl_model_state learning_node<feat,lb>::get_drift(){
	// Getting the drift vector is done as getting the local statistic
	return dl_model_state(_learner->Parameters(), _learner->getNumOfUpdates());
}

template<typename feat,typename lb>
void learning_node<feat,lb>::set_drift(tensor_message mdl){
//void learning_node<feat,lb>::set_drift(dl_model_state mdl){
	// Update the local learner with the model sent by the coordinator
	_learner->update_model(mdl._model);
}

template<typename feat,typename lb>
void learning_node<feat,lb>::warmup(std::vector<matrix<feature_t>>& batch, std::vector<label_t>& labels){
	_learner->fit(batch,labels);
}

template<typename feat,typename lb>
void learning_node<feat,lb>::end_warmup(){
	coord.end_warmup(this);
}

template<typename feat,typename lb>
void learning_node<feat,lb>::update_stream(std::vector<matrix<feature_t>>& batch, std::vector<label_t>& labels){
	_learner->fit(batch,labels);
	datapoints_seen += batch.size();
	
	if( dl_safezone_function* entity = dynamic_cast<Param_Variance_safezone_func*>(szone.getSZone()) ){
		if(szone(datapoints_seen) <= 0.){
			datapoints_seen = 0;
			if(szone(_learner->Parameters())<0.){
				coord.local_violation(this);
			}
		}
	}else{
		if(szone(datapoints_seen) <= 0.)
			coord.local_violation(this);
	}
}

template<typename feat,typename lb>
void learning_node<feat,lb>::initializeLearner(){
	if (cfg().learning_algorithm == "LeNet"){
		_learner = new LeNet<feature_t,label_t>();
	}
}

template<typename feat,typename lb>
void learning_node<feat,lb>::setup_connections(){
	num_sites = coord.proc()->k;
}

template<typename feat,typename lb>
struct learning_node_proxy : remote_proxy< learning_node<feat,lb> >
{
	typedef learning_node<feat,lb> node_t;
	REMOTE_METHOD(node_t, reset);
	REMOTE_METHOD(node_t, get_drift);
	REMOTE_METHOD(node_t, set_drift);
	learning_node_proxy(process* p) : remote_proxy<node_t>(p) {}
};
	
} // end namespace DL_gm_networks



} // ml_gm_proto

namespace dds {
	template<>
	inline size_t byte_size<ml_gm_proto::DL_gm_networks::learning_node<unsigned char, unsigned long>*>(ml_gm_proto::DL_gm_networks::learning_node<unsigned char, unsigned long>* const &) { return 4; }
}

#endif
