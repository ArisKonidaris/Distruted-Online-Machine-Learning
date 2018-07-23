#ifndef __DL_FGM_NETWORKS_HH__
#define __DL_FGM_NETWORKS_HH__

#include <boost/range/adaptors.hpp>
#include <boost/shared_ptr.hpp>
#include <random>
#include <fstream>
#include <iostream>

#include "ML_GM_Proto.hh"

namespace ml_gm_proto {
	
using namespace dds;
using namespace H5;
using namespace data_src;
using std::map;
using std::cout;
using std::endl;

namespace DL_fgm_networks {

using namespace ml_gm_proto::DLib_GM_Proto;

template<typename feat, typename lb>
struct coordinator;
template<typename feat, typename lb>
struct learning_node;
template<typename feat, typename lb>
struct learning_node_proxy;

template<typename feat,typename lb>
struct FGM_Net : dl_gm_learning_network<feat, lb, FGM_Net, coordinator, learning_node>{	
	FGM_Net(const set<source_id>& _hids, const string& _name, dl_continuous_query<feat,lb>* _Q);
};

template<typename feat,typename lb>
FGM_Net<feat,lb>::FGM_Net(const set<source_id>& _hids, const string& _name, dl_continuous_query<feat,lb>* _Q)
: dl_gm_learning_network<feat, lb, FGM_Net, coordinator, learning_node>(_hids, _name, _Q){
	this->set_protocol_name("LD_FGM");
}

/**
	This is a hub implementation for the Functional Geometric Method protocol.

 */
template<typename feat,typename lb>
struct coordinator : process
{
	typedef feat feature_t;
	typedef lb label_t;
	typedef coordinator<feature_t,label_t> coordinator_t;
	typedef learning_node<feature_t,label_t> node_t;
	typedef learning_node_proxy<feature_t,label_t> node_proxy_t;
	typedef FGM_Net<feature_t,label_t> network_t;

	proxy_map<node_proxy_t, node_t> proxy;

	//
	// protocol stuff
	//
	DLIB_Learner<feat,lb>* global_learner;
    dl_continuous_query<feature_t,label_t>* Q;   // continuous query
    dl_query_state* query; 		                 // current query state
 	dl_safezone_function* safe_zone; 	         // the safe zone wrapper

	size_t k;					                 // number of sites

	// index the nodes
	map<node_t*, size_t> node_index;
	map<node_t*, size_t> node_bool_drift;
	vector<node_t*> node_ptr;
	
	float phi;                          // The phi value of the functional geometric protocol.
	float quantum;                      // The quantum of the functional geometric protocol.
	int counter;                        // A counter used by the functional geometric protocol.
	double barrier;                     // The smallest number the zeta function can reach.
	int cnt;                            // Helping counter.
	
	vector<resizable_tensor*> Params;   // A placeholder for the parameters send by the nodes.
	vector<resizable_tensor*> Beta;     // The beta vector used by the protocol for the rebalancing process.

	coordinator(network_t* nw, dl_continuous_query<feature_t, label_t>* _Q); 
	~coordinator();

	inline network_t* net() { return static_cast<network_t*>(host::net()); }
	inline const dl_protocol_config& cfg() const { return Q->config; }

	// Initialize the Learner and its' variables.
	void initializeLearner();

	// Method used by the hub to establish the connections with the nodes of the star network.
	void setup_connections() override;

	// initialize a new round
	void start_round();
	void finish_round();
	void finish_rounds();
	
	// Rebalancing method.
	void Rebalance();

	// Getting the model of a node.
	void fetch_updates(node_t* n);
	
	// Printing and saving the accuracy.
	void Progress();
	
	// Get the communication statistics of experiment.
	vector<size_t> Statistics() const;
	
	// Warming up the coordinator.
	void warmup(std::vector<matrix<feat>>& batch, std::vector<lb>& labels);
	
	// Ending the warmup of the network.
	void end_warmup();

	// remote call on host violation
	oneway send_increment(increment inc);

	// statistics
	size_t num_rounds;				 // total number of rounds
	size_t num_subrounds;			 // total number of subrounds
	size_t num_rebalances;           // total number of rebalances
	size_t sz_sent;					 // total safe zones sent
 	size_t total_updates;		     // number of stream updates received

};

template<typename feat,typename lb>
void coordinator<feat,lb>::start_round(){
	
	// Resets.
	cnt = 0;
	counter = 0;
	for(auto layer : Beta){
		//dlib::cpu::affine_transform(*layer, *layer, 0., 0.);
		dlib::cuda::affine_transform(*layer, *layer, 0.);
	}
	
	// Calculating the new phi, quantum and the minimum acceptable value for phi.
	phi = k*safe_zone->Zeta(query->GlobalModel);
	quantum = (float)(phi/(2*k));
	barrier = cfg().precision*phi;
	
	for(auto n : net()->sites) {
		sz_sent ++;
		proxy[n].reset(dl_safezone(safe_zone), float_value(quantum));
		node_bool_drift[n] = 0;
	}
	
	num_rounds++;
	num_subrounds++;
}

template<typename feat,typename lb>
oneway coordinator<feat,lb>::send_increment(increment inc){
	counter += inc.increase;
	if(counter>k){
		phi = 0.;
		// Collect all data
		for(auto n : node_ptr) {
			phi += proxy[n].get_zed_value().value;
		}
		if(phi>=barrier){
			counter = 0;
			quantum = (float)(phi/(2*k));
			// send the new quantum
			for(auto n : node_ptr) {
				proxy[n].take_quantum(float_value(quantum));
			}
			num_subrounds++;
		}else{
			for(auto n : node_ptr) {
				fetch_updates(n);
			}
			Rebalance();
		}
	}
}

template<typename feat,typename lb>
void coordinator<feat,lb>::fetch_updates(node_t* node){
	dl_model_state<resizable_tensor> up = proxy[node].get_drift();
	if( std::abs(safe_zone->checkAdmissibleNorm(up._model)-std::sqrt(safe_zone->hyperparameters.at(0))) > 1e-6 ){
		if(node_bool_drift[node]==0){
			node_bool_drift[node] = 1;
			cnt++;
		}
		for(size_t i=0; i<up._model.size(); i++){
			//dlib::cpu::add(*Params.at(i), *Params.at(i), *up._model.at(i));
			dlib::cuda::add(1., *Params.at(i), 1., *up._model.at(i));
		}
	}
	total_updates += up.updates;
}

// Rebalancing attempt.
template<typename feat,typename lb>
void coordinator<feat,lb>::Rebalance() {
	
	for(size_t i=0;i<Beta.size();i++){
		//dlib::cpu::add(*Beta.at(i), *Beta.at(i), *Params.at(i));
		dlib::cuda::affine_transform(*Beta.at(i), *Beta.at(i), *Params.at(i), 1., 1.);
	}
	for(size_t i=0;i<Beta.size();i++){
		//dlib::cpu::affine_transform(*Params.at(i), *query->GlobalModel.at(i), *Beta.at(i), 1., 2./cnt, 0.);
		dlib::cuda::affine_transform(*Params.at(i), *query->GlobalModel.at(i), *Beta.at(i), 1., 2./cnt);
	}
	//phi = (k/2)*safe_zone->Zeta(Params) + (k/2)*safe_zone->Zeta(query->GlobalModel);
	phi = (cnt/2)*safe_zone->Zeta(Params) + ( (cnt/2)+(k-cnt) )*safe_zone->Zeta(query->GlobalModel);
	for(size_t i=0;i<Params.size();i++){
		//dlib::cpu::affine_transform(*Params.at(i), *Params.at(i), 0., 0.);
		dlib::cuda::affine_transform(*Params.at(i), *Params.at(i), 0.);
	}
	
	if (phi>=cfg().reb_mult*barrier){
		counter = 0;
		quantum = (float)(phi/(2*k));
		// send the new quantum
		for(auto n : node_ptr) {
			proxy[n].rebalance(float_value(quantum));
		}
		num_subrounds++;
		num_rebalances++;
	}else{
		finish_round();
	}
}

// initialize a new round
template<typename feat,typename lb>
void coordinator<feat,lb>::finish_round() {
	// New round
	for(size_t i=0;i<Beta.size();i++){
		//dlib::cpu::affine_transform(*Beta.at(i), *query->GlobalModel.at(i), *Beta.at(i), 1., std::pow(cnt,-1), 0.);
		dlib::cuda::affine_transform(*Beta.at(i), *query->GlobalModel.at(i), *Beta.at(i), 1., std::pow(cnt,-1));
	}
	query->update_estimate(Beta);
	global_learner->update_model(Beta);
	start_round();
}

template<typename feat,typename lb>
void coordinator<feat,lb>::finish_rounds(){
	
	cout << endl;
	cout << "Global model of network " << net()->name() << "." << endl;
	cout << "tests : " << Q->testSet->size() << endl;
	
	// Query thr accuracy of the global model.
	query->accuracy = Q->queryAccuracy(global_learner);
	
	// See the total number of points received by all the nodes. For debugging.
	for(auto n:node_ptr){
		size_t updates = n->_learner->getNumOfUpdates();
		total_updates += updates;
	}
	
	// Print the results.
	if(Q->config.learning_algorithm == "LeNet"){
		cout << "accuracy : " << std::setprecision(6) << (float)100.*query->accuracy << "%" << endl;
	}else{
		cout << "Another measure of accuracy for later on." << endl;
	}
	cout << "Number of rounds : " << num_rounds << endl;
	cout << "Number of subrounds : " << num_subrounds << endl;
	cout << "Number of rebalances : " << num_rebalances << endl;
	cout << "Total updates : " << total_updates << endl;
	
}

template<typename feat,typename lb>
void coordinator<feat,lb>::Progress(){
	
	cout << "Global model of network " << net()->name() << "." << endl;

	// Query the accuracy of the global model.
	if(Q->config.learning_algorithm == "LeNet"){
		query->accuracy = Q->queryAccuracy(global_learner);
	}
	
	cout << "accuracy : " << std::setprecision(6) << (float)100.*query->accuracy << "%" << endl;
	cout << "Number of rounds : " << num_rounds << endl;
	cout << "Number of subrounds : " << num_subrounds << endl;
	cout << "Number of rebalances : " << num_rebalances << endl;
	cout << "Total updates : " << total_updates << endl;
	cout << endl;
}

template<typename feat,typename lb>
vector<size_t>  coordinator<feat,lb>::Statistics() const{
	vector<size_t> stats;
	stats.push_back(num_rounds);
	stats.push_back(num_subrounds);
	stats.push_back(num_rebalances);
	stats.push_back(sz_sent);
	return stats;
}

template<typename feat,typename lb>
void coordinator<feat,lb>::warmup(std::vector<matrix<feat>>& batch, std::vector<lb>& labels){
	global_learner->fit(batch,labels);
	total_updates+=batch.size();
	if(query->GlobalModel.size()==0){
		query->initializeGlobalModel(global_learner->Parameters());
		for(auto layer:global_learner->Parameters()){
			resizable_tensor* l;
			resizable_tensor* _l;
			l = new resizable_tensor();
			_l = new resizable_tensor();
			l->set_size(layer->num_samples(),layer->k(),layer->nr(),layer->nc());
			_l->set_size(layer->num_samples(),layer->k(),layer->nr(),layer->nc());
			*l = 0.;
			*_l = 0.;
			Beta.push_back(l);
			Params.push_back(_l);
		}
	}
}

template<typename feat,typename lb>
void coordinator<feat,lb>::end_warmup(){
	for(size_t i=0;i<Beta.size();i++){
		//dlib::cpu::affine_transform(*Beta.at(i), *global_learner->Parameters().at(i), 1., 0.);
		dlib::cuda::affine_transform(*Beta.at(i), *global_learner->Parameters().at(i), 1.);
	}
	query->update_estimate(Beta);
	start_round();
}
	
template<typename feat,typename lb>
void coordinator<feat,lb>::initializeLearner(){
	if (cfg().learning_algorithm == "LeNet"){
		global_learner = new LeNet<feature_t,label_t>(this->name());
		
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
	cout << "Synchronizer " << this->name() << " initialized its network." << endl;
}

template<typename feat,typename lb>
void coordinator<feat,lb>::setup_connections(){
	using boost::adaptors::map_values;
	proxy.add_sites(net()->sites);
	for(auto n : net()->sites) {
		node_index[n] = node_ptr.size();
		node_bool_drift[n] = 0;
		node_ptr.push_back(n);
	}
	k = node_ptr.size();
}

template<typename feat,typename lb>
coordinator<feat,lb>::coordinator(network_t* nw, dl_continuous_query<feat,lb>* _Q)
: 	process(nw), proxy(this),
	Q(_Q),
	k(0),
	num_rounds(0), num_subrounds(0),
	sz_sent(0), total_updates(0), num_rebalances(0){
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
	REMOTE_METHOD(coordinator_t, send_increment);
	coord_proxy(process* c) : remote_proxy<coordinator_t>(c) { } 
};


/**
	This is a site implementation for the Functional Geometric Method protocol.

 */
template<typename feat,typename lb>
struct learning_node : local_site {

	typedef feat feature_t;
	typedef lb label_t;
	typedef coordinator<feature_t, label_t> coordinator_t;
	typedef learning_node<feature_t, label_t> node_t;
	typedef learning_node_proxy<feature_t, label_t> node_proxy_t;
	typedef FGM_Net<feature_t, label_t> network_t;
	typedef coord_proxy<feature_t, label_t> coord_proxy_t;
    typedef dl_continuous_query<feature_t, label_t> continuous_query_t;

    continuous_query_t* Q;               // The query management object.
    dl_safezone szone;                   // The safezone object.
	DLIB_Learner<feat,lb>* _learner;
	vector<resizable_tensor*> Delta_Vector;
	vector<resizable_tensor*> E_Delta;

	int num_sites;			             // Number of sites.
	coord_proxy_t coord;                 // The proxy of the coordinator/hub.
	
	int counter;                         // The counter used by the FGM protocol.
	float quantum;                       // The quantum provided by the hub.
	float zeta;                          // The value of the safezone function.

	learning_node(network_t* net, source_id hid, continuous_query_t* _Q)
	:	local_site(net, hid), Q(_Q), coord( this )
	{ 
		coord <<= net->hub;
		initializeLearner();
	};

	inline const dl_protocol_config& cfg() const { return Q->config; }

	void initializeLearner();

	void setup_connections() override;

	void update_stream(std::vector<matrix<feature_t>>& batch, std::vector<label_t>& labels);

	//
	// Remote methods
	//

	// called at the start of a round
	oneway reset(const dl_safezone& newsz, const float_value qntm); 
	
	// Refreshing the quantum for a new subround
	oneway take_quantum(const float_value qntm);
	
	// Refreshing the quantum and reverting the drift vector back to E
	oneway rebalance(const float_value qntm);

	// transfer data to the coordinator
	dl_model_state<resizable_tensor> get_drift();
	
	// Transfer the value of z(Xi) to the coordinator
	float_value get_zed_value();
	
};

template<typename feat,typename lb>
oneway learning_node<feat,lb>::reset(const dl_safezone& newsz, const float_value qntm){
	counter = 0;
	szone = newsz;                                                    // Reset the safezone object
	quantum = 1.*qntm.value;                                          // Reset the quantum
	_learner->update_model(szone.getSZone()->getGlobalModel());       // Updates the parameters of the local learner
	zeta = szone(_learner->Parameters(), true);                       // Reset zeta
	
	// TODO: Make a pretty function for it.
	for (size_t i=0; i<_learner->Parameters().size(); i++){
		//dlib:memcpy(*E_Delta.at(i), *_learner->Parameters().at(i));
		//dlib::cpu::affine_transform(*E_Delta.at(i), *_learner->Parameters().at(i), 1., 0.);
		dlib::cuda::affine_transform(*E_Delta.at(i), *_learner->Parameters().at(i), 1.);
	}
}


template<typename feat,typename lb>
oneway learning_node<feat,lb>::take_quantum(const float_value qntm){
	counter = 0;                                                // Reset counter
	quantum = 1.*qntm.value;                                    // Update the quantum
	zeta = szone(_learner->Parameters(), E_Delta);              // Update zeta
}

template<typename feat,typename lb>
oneway learning_node<feat,lb>::rebalance(const float_value qntm){
	counter = 0;
	quantum = 1.*qntm.value;                                  // Update the quantum
	
	for (size_t i=0; i<_learner->Parameters().size(); i++){
		//dlib::cpu::affine_transform(*E_Delta.at(i), *_learner->Parameters().at(i), 1., 0.);
		dlib::cuda::affine_transform(*E_Delta.at(i), *_learner->Parameters().at(i), 1.);
	}
	
	zeta = szone(_learner->Parameters(), E_Delta);            // Reset zeta
}

template<typename feat,typename lb>
dl_model_state<resizable_tensor> learning_node<feat,lb>::get_drift(){
	// Getting the drift vector is done as getting the local statistic
	for (size_t i=0; i<_learner->Parameters().size(); i++){
		//dlib::cpu::affine_transform(*Delta_Vector.at(i), *_learner->Parameters().at(i), *E_Delta.at(i), 1., -1., 0.);
		dlib::cuda::affine_transform(*Delta_Vector.at(i), *_learner->Parameters().at(i), *E_Delta.at(i), 1., -1.);
	}
	return dl_model_state<resizable_tensor>(Delta_Vector, _learner->getNumOfUpdates());
}

template<typename feat,typename lb>
float_value learning_node<feat,lb>::get_zed_value(){
	return float_value((float)szone(_learner->Parameters(), E_Delta));
}

template<typename feat,typename lb>
void learning_node<feat,lb>::update_stream(std::vector<matrix<feature_t>>& batch, std::vector<label_t>& labels){
	_learner->fit(batch,labels);
	if( dl_safezone_function* entity = dynamic_cast<Param_Variance_safezone_func*>(szone.getSZone()) ){
		int c_now = std::floor((zeta-szone(_learner->Parameters(), E_Delta))/quantum);
		if(c_now-counter>0){
			coord.send_increment(increment(c_now-counter));
			counter = c_now;
		}
	}
}

template<typename feat,typename lb>
void learning_node<feat,lb>::initializeLearner(){
	if (cfg().learning_algorithm == "LeNet"){
		_learner = new LeNet<feature_t,label_t>(this->name());
		
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
		_learner->fit(random_image_init,random_label_init);
		_learner->getNumOfUpdates();
		
		// Initializing the Delta_Vector and E_Drift vectors.
		for(auto layer:_learner->Parameters()){
			resizable_tensor* l;
			resizable_tensor* _l;
			l = new resizable_tensor();
			_l = new resizable_tensor();
			l->set_size(layer->num_samples(),layer->k(),layer->nr(),layer->nc());
			_l->set_size(layer->num_samples(),layer->k(),layer->nr(),layer->nc());
			*l = 0.;
			*_l = 0.;
			Delta_Vector.push_back(l);
			E_Delta.push_back(_l);
		}
	}
	cout << "Local site " << this->name() << " initialized its network." << endl;
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
	REMOTE_METHOD(node_t, take_quantum);
	REMOTE_METHOD(node_t, rebalance);
	REMOTE_METHOD(node_t, get_drift);
	REMOTE_METHOD(node_t, get_zed_value);
	learning_node_proxy(process* p) : remote_proxy<node_t>(p) {}
};
	
} // end namespace DL_fgm_networks

} // ml_gm_proto

namespace dds {
	template<>
	inline size_t byte_size<ml_gm_proto::DL_fgm_networks::learning_node<unsigned char, unsigned long>*>(ml_gm_proto::DL_fgm_networks::learning_node<unsigned char, unsigned long>* const &) { return 4; }
}

#endif
