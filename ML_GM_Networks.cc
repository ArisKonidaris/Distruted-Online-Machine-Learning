#include "ML_GM_Networks.hh"

using namespace ml_gm_proto;
using namespace ml_gm_proto::ML_gm_networks;


/*********************************************
	node
*********************************************/

oneway learning_node::reset(const safezone& newsz){
	szone = newsz;       // Reset the safezone object
	datapoints_seen = 0; // Reset the drift vector
	_learner->update_model_by_ref(szone.getSZone()->getGlobalModel()); // Updates the parameters of the local learner
}

model_state learning_node::get_drift(){
	// Getting the drift vector is done as getting the local statistic
	return model_state(_learner->getModel(), _learner->getNumOfUpdates());
}

void learning_node::set_drift(model_state mdl){
	// Update the local learner with the model sent by the coordinator
	_learner->update_model_by_ref(mdl._model);
}

void learning_node::update_stream(arma::mat& batch, arma::mat& labels){
	_learner->fit(batch,labels);
	datapoints_seen += batch.n_cols;
	
	if( ml_safezone_function* entity = dynamic_cast<Variance_safezone_func*>(szone.getSZone()) ){
		if(szone(datapoints_seen) <= 0.){
			datapoints_seen = 0;
			if(szone(_learner->getModel())<0.){
				coord.local_violation(this);
			}
		}
	}else{
		if(szone(datapoints_seen) <= 0.)
			coord.local_violation(this);
	}
	
}

void learning_node::initializeLearner(){
	if (cfg().learning_algorithm == "PA"){
		_learner = new PassiveAgressiveClassifier(cfg().cfgfile, cfg().network_name);
	}else if(cfg().learning_algorithm == "MLP"){
		_learner = new MLP_Classifier(cfg().cfgfile, cfg().network_name);
	}else if(cfg().learning_algorithm == "PA_Reg"){
		_learner = new PassiveAgressiveRegression(cfg().cfgfile, cfg().network_name);
	}else if(cfg().learning_algorithm == "NN_Reg"){
		_learner = new NN_Regressor(cfg().cfgfile, cfg().network_name);
	}
	_learner->initializeModel(Q->testSet->n_rows);
}

void learning_node::setup_connections(){
	num_sites = coord.proc()->k;
}


/*********************************************
	coordinator
*********************************************/

void coordinator::start_round(){
	for(auto n : net()->sites) {
		sz_sent ++;
		proxy[n].reset(safezone(safe_zone));
	}
	num_rounds++;
}

oneway coordinator::local_violation(sender<node_t> ctx){
	
	node_t* n = ctx.value;
	num_violations++;
	
	B.clear(); // Clear the balanced nodes set.
	Mean.zeros(); // Clear the mean global model.
	
	if( ml_safezone_function* entity = dynamic_cast<Batch_Learning*>(safe_zone) ){
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

void coordinator::fetch_updates(node_t* node){
	model_state up = proxy[node].get_drift();
	Mean += up._model;	
	total_updates += up.updates;
}

// initialize a new round
void coordinator::finish_round() {

	// Collect all data
	//for(auto n : Bcompl) {
	for(auto n : node_ptr) {
		fetch_updates(n);
	}
	Mean /= (double)k;

	// New round
	query->update_estimate(Mean);
	global_learner->update_model_by_ref(Mean);
	
	start_round();

}

void coordinator::rebalance_set(){
	
	
}

void coordinator::Kamp_Rebalance(node_t* lvnode){
	
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
		if(safe_zone->checkIfAdmissible( Mean/(double)B.size() ) > 0. || B.size() == k )
			break;
	}
	
	Mean /=(double)B.size();
	
	if(B.size() < k){ 
		// Rebalancing
		for(auto n : B) {
			proxy[n].set_drift(model_state(Mean, 0));
		}
		num_subrounds++;
	}else{
		// New round
		num_violations = 0;
		query->update_estimate(Mean);
		global_learner->update_model_by_ref(Mean);
		start_round();
	}
	
}

void coordinator::finish_rounds(){
	
	cout << endl;
	cout << "Global model of network " << net()->name() << "." << endl;
	cout << "tests : " << Q->testSet->n_cols << endl;
	
	// Query thr accuracy of the global model.
	query->accuracy = Q->queryAccuracy(global_learner);
	
	// See the total number of points received by all the nodes. For debugging.
	for(auto nd:node_ptr){
		total_updates += nd->_learner->getNumOfUpdates();
	}
	
	// Print the results.
	if(Q->config.learning_algorithm == "PA"
	|| Q->config.learning_algorithm == "MLP"){
		cout << "accuracy : " << std::setprecision(6) << query->accuracy << "%" << endl;
	}else{
		cout << "accuracy : " << std::setprecision(6) << query->accuracy << endl;
	}
	cout << "Number of rounds : " << num_rounds << endl;
	cout << "Number of subrounds : " << num_subrounds << endl;
	cout << "Total updates : " << total_updates << endl;
	
}

void coordinator::warmup(arma::mat& batch, arma::mat& labels){
	global_learner->fit(batch,labels);
	if(query->GlobalModel.n_elem==0){
		query->initializeGlobalModel(global_learner->modelDimensions());
		Mean = arma::mat(global_learner->modelDimensions(), arma::fill::zeros);
	}
	query->update_estimate(global_learner->getModel());
	total_updates+=batch.n_cols;
}

void coordinator::setup_connections(){
	using boost::adaptors::map_values;
	proxy.add_sites(net()->sites);
	for(auto n : net()->sites) {
		node_index[n] = node_ptr.size();
		node_ptr.push_back(n);
	}
	k = node_ptr.size();
}

void coordinator::initializeLearner(){
	if (cfg().learning_algorithm == "PA"){
		global_learner = new PassiveAgressiveClassifier(cfg().cfgfile, cfg().network_name);
	}else if (cfg().learning_algorithm == "MLP"){
		global_learner = new MLP_Classifier(cfg().cfgfile, cfg().network_name);
	}else if(cfg().learning_algorithm == "PA_Reg"){
		global_learner = new PassiveAgressiveRegression(cfg().cfgfile, cfg().network_name);
	}else if(cfg().learning_algorithm == "NN_Reg"){
		global_learner = new NN_Regressor(cfg().cfgfile, cfg().network_name);
	}
	global_learner->initializeModel(Q->testSet->n_rows);
}

coordinator::coordinator(network_t* nw, continuous_query* _Q)
: 	process(nw), proxy(this),
	Q(_Q),
	k(0),
	num_violations(0), num_rounds(0), num_subrounds(0),
	sz_sent(0), total_updates(0){
	initializeLearner();
	query = Q->create_query_state();
	safe_zone = query->safezone(cfg().cfgfile, cfg().distributed_learning_algorithm);
}

coordinator::~coordinator(){
	delete safe_zone;
	delete query;
}

ML_gm_networks::network::network(const set<source_id>& _hids, const string& _name, continuous_query* _Q)
: gm_learning_network_t(_hids, _name, _Q){
	this->set_protocol_name("GM");
}