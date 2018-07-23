#ifndef __ML_FGM_NETWORKS_HH__
#define __ML_FGM_NETWORKS_HH__

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

namespace ML_fgm_networks {
	
using namespace ml_gm_proto::MlPack_GM_Proto;

struct node_proxy;
struct coordinator;
struct learning_node;
struct learning_node_proxy;

struct FGM_Net : gm_learning_network<FGM_Net, coordinator, learning_node>
{
	typedef gm_learning_network<network_t, coordinator_t, node_t> gm_learning_network_t;
	
	FGM_Net(const set<source_id>& _hids, const string& _name, continuous_query* _Q);
};

/**
	This is a hub implementation for the Functional Geometric Method protocol.

 */
struct coordinator : process
{
	typedef coordinator coordinator_t;
	typedef learning_node node_t;
	typedef learning_node_proxy node_proxy_t;
	typedef FGM_Net network_t;

	proxy_map<node_proxy_t, node_t> proxy;

	//
	// protocol stuff
	//
	MLPACK_Learner* global_learner;
    continuous_query* Q;   		              // continuous query
    query_state* query; 		              // current query state
 	ml_safezone_function* safe_zone; 	      // the safe zone wrapper

	size_t k;					              // number of sites

	// index the nodes
	map<node_t*, size_t> node_index;
	map<node_t*, size_t> node_bool_drift;
	vector<node_t*> node_ptr;
	
	float phi;                       // The phi value of the functional geometric protocol.
	float quantum;                   // The quantum of the functional geometric protocol.
	int counter;                     // A counter used by the functional geometric protocol.
	float barrier;                  // The smallest number the zeta function can reach.
	int cnt;                         // Helping counter.
	
	vector<arma::mat> Params;        // A placeholder for the parameters send by the nodes.
	vector<arma::mat> Beta;          // The beta vector used by the protocol for the rebalancing process.    

	coordinator(network_t* nw, continuous_query* _Q); 
	~coordinator();

	inline network_t* net() { return static_cast<network_t*>(host::net()); }
	inline const protocol_config& cfg() const { return Q->config; }

	// Initialize the Learner and its' variables.
	void initializeLearner();

	// Method used by the hub to establish the connections with the nodes of the star network.
	void setup_connections() override;

	// Method loading the warmup dataset.
	void warmup(arma::mat& batch, arma::mat& labels);
	
	// Ending the warmup of the network.
	void end_warmup();

	// initialize a new round.
	void start_round();
	void finish_round();
	void finish_rounds();
	
	// Rebalancing method.
	void Rebalance();
	
	// Getting the model of a node.
	void fetch_updates(node_t* n);

	// Printing and saving the accuracy.
	void Progress();
	
	// Getting the accuracy of the global learner.
	double getAccuracy();
	
	// Get the communication statistics of experiment.
	vector<size_t> Statistics() const;
	
	// remote call on host violation
	oneway send_increment(increment inc);
	
	// Methods used in case of concept drift.
	matrix_message hybrid_drift(sender<node_t> ctx, int_num rows, int_num cols);
	matrix_message virtual_drift(sender<node_t> ctx, int_num rows);
	oneway real_drift(sender<node_t> ctx, int_num cols);
	
	// statistics
	size_t num_rounds;				 // total number of rounds
	size_t num_subrounds;			 // total number of subrounds
	size_t num_rebalances;           // total number of rebalances
	size_t sz_sent;					 // total safe zones sent
 	size_t total_updates;		     // number of stream updates received

};

struct coord_proxy : remote_proxy< coordinator >
{
	using coordinator_t = coordinator;
	REMOTE_METHOD(coordinator_t, send_increment);
	REMOTE_METHOD(coordinator_t, hybrid_drift);
	REMOTE_METHOD(coordinator_t, virtual_drift);
	REMOTE_METHOD(coordinator_t, real_drift);
	coord_proxy(process* c) : remote_proxy<coordinator_t>(c) { } 
};

/**
	This is a site implementation for the Functional Geometric Method protocol.

 */
struct learning_node : local_site {

	typedef coordinator coordinator_t;
	typedef learning_node node_t;
	typedef learning_node_proxy node_proxy_t;
	typedef FGM_Net network_t;
	typedef coord_proxy coord_proxy_t;
    typedef continuous_query continuous_query_t;

    continuous_query* Q;                 // The query management object.
    safezone szone;                      // The safezone object.
	MLPACK_Learner* _learner;            // The learning algorithm.
	vector<arma::mat> Delta_Vector;
	vector<arma::mat> E_Delta;

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

	inline const protocol_config& cfg() const { return Q->config; }

	void initializeLearner();

	void setup_connections() override;

	void update_stream(arma::mat& batch, arma::mat& labels);
	
	void update_drift(vector<arma::mat*>& params);

	//
	// Remote methods
	//

	// called at the start of a round
	oneway reset(const safezone& newsz, const float_value qntm);
	
	// Refreshing the quantum for a new subround
	oneway take_quantum(const float_value qntm);
	
	// Refreshing the quantum and reverting the drift vector back to E
	oneway rebalance(const float_value qntm);
	
	// Transfer data to the coordinator
	model_state get_drift();
	
	// Transfer the value of z(Xi) to the coordinator
	float_value get_zed_value();
	
	// Transfering the new random paramenters of the model generated by the hub.
	oneway augmentAlphaMatrix(matrix_message params);
	
	// Signaling for the augmention of the beta vector.
	oneway augmentBetaMatrix(int_num cols);
	
	// Loading the parameters of the hub to the local model.
	oneway set_HStatic_variables(const p_model_state& SHParams);
	
};

struct learning_node_proxy : remote_proxy< learning_node >
{
	typedef learning_node node_t;
	REMOTE_METHOD(node_t, reset);
	REMOTE_METHOD(node_t, take_quantum);
	REMOTE_METHOD(node_t, rebalance);
	REMOTE_METHOD(node_t, get_drift);
	REMOTE_METHOD(node_t, get_zed_value);
	REMOTE_METHOD(node_t, set_HStatic_variables);
	REMOTE_METHOD(node_t, augmentAlphaMatrix);
	REMOTE_METHOD(node_t, augmentBetaMatrix);
	learning_node_proxy(process* p) : remote_proxy<node_t>(p) {}
};
	
} // end namespace ML_fgm_networks

} // ml_gm_proto

namespace dds {
	template<>
	inline size_t byte_size<ml_gm_proto::ML_fgm_networks::learning_node*>(ml_gm_proto::ML_fgm_networks::learning_node* const &) { return 4; }
}

#endif