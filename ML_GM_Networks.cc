#include "ML_GM_Networks.hh"

using namespace ml_gm_proto;
using namespace ml_gm_proto::ML_gm_networks;


/*********************************************
	coordinator
*********************************************/

void coordinator::start_round(){
	// Send new safezone.
	for(auto n : net()->sites) {
		if (cfg().learning_algorithm=="ELM" && num_rounds==0){
			proxy[n].set_HStatic_variables(p_model_state(global_learner->getHModel(), 0));
		}
		sz_sent ++;
		proxy[n].reset(safezone(safe_zone));
	}
	num_rounds++;
}

oneway coordinator::local_violation(sender<node_t> ctx){
	
	node_t* n = ctx.value;
	num_violations++;
	
	// Clear
	B.clear(); // Clear the balanced nodes set.
	for(size_t i=0; i<Mean.size(); i++){
		Mean.at(i).zeros();
	}
	cnt = 0;
	
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
	if( !arma::approx_equal(arma::mat(arma::size(up._model.at(0)),arma::fill::zeros), up._model.at(0), "absdiff", 1e-6) ){
		cnt++;
		for(size_t i=0; i<up._model.size(); i++){
			Mean.at(i) += up._model.at(i);
		}
	}
	total_updates += up.updates;
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
		for(size_t i=0; i<Mean.size(); i++)
			Mean.at(i) /= cnt;
		if(safe_zone->checkIfAdmissible_v2(Mean) > 0. || B.size() == k )
			break;
		for(size_t i=0; i<Mean.size(); i++)
			Mean.at(i) *= cnt;	
	}
	
	if(B.size() < k){ 
		// Rebalancing
		for(size_t i=0; i<Mean.size(); i++)
			Mean.at(i) += query->GlobalModel.at(i);	
		for(auto n : B) {
			proxy[n].set_drift(model_state(Mean, 0));
		}
		num_subrounds++;
	}else{
		// New round
		num_violations = 0;
		query->update_estimate_v2(Mean);
		global_learner->update_model(query->GlobalModel);
		start_round();
	}
	
}

// initialize a new round
void coordinator::finish_round() {

	// Collect all data
	for(auto n : node_ptr) {
		fetch_updates(n);
	}
	for(size_t i=0; i<Mean.size(); i++)
		Mean.at(i) /= cnt;

	// New round
	query->update_estimate_v2(Mean); 
	global_learner->update_model(query->GlobalModel);
	
	start_round();

}

matrix_message coordinator::hybrid_drift(sender<node_t> ctx, int_num rows, int_num cols){
	node_t* n = ctx.value;
	
	//Virtual Drift. Updating hidden layer matrix.
	global_learner->handleVD(rows.number);
	arma::mat new_hidden = arma::zeros<arma::mat>(rows.number, global_learner->getHModel().at(0)->n_cols);
	new_hidden += global_learner->getHModel().at(0)->rows(global_learner->getHModel().at(0)->n_rows-rows.number,
													      global_learner->getHModel().at(0)->n_rows-1);
	for(auto st : net()->sites){
		if(st!=n){
			proxy[st].augmentAlphaMatrix(matrix_message(new_hidden));
		}
	}
	
	// Real Drift. Updating beta vector.
	global_learner->handleRD(cols.number);
	for(auto st : net()->sites){
		if(st!=n){
			proxy[st].augmentBetaMatrix(cols);
		}
	}
	Mean.at(1).resize(Mean.at(1).n_rows, Mean.at(1).n_cols+cols.number);
	query->GlobalModel.at(1) = arma::mat(arma::size(*global_learner->getModel().at(1)), arma::fill::zeros);
	query->GlobalModel.at(1) += *global_learner->getModel().at(1);
	
	return matrix_message(new_hidden);
}

matrix_message coordinator::virtual_drift(sender<node_t> ctx, int_num rows){
	// Updating hidden layer matrix.
	node_t* n = ctx.value;
	global_learner->handleVD(rows.number);
	arma::mat new_hidden = arma::zeros<arma::mat>(rows.number, global_learner->getHModel().at(0)->n_cols);
	new_hidden += global_learner->getHModel().at(0)->rows(global_learner->getHModel().at(0)->n_rows-rows.number,
													      global_learner->getHModel().at(0)->n_rows-1);
	for(auto st : net()->sites){
		if(st!=n){
			proxy[st].augmentAlphaMatrix(matrix_message(new_hidden));
		}
	}
	return matrix_message(new_hidden);
}

oneway coordinator::real_drift(sender<node_t> ctx, int_num cols){
	// Updating beta vector.
	node_t* n = ctx.value;
	global_learner->handleRD(cols.number);
	for(auto st : net()->sites){
		if(st!=n){
			proxy[st].augmentBetaMatrix(cols);
		}
	}
	Mean.at(1).resize(Mean.at(1).n_rows, Mean.at(1).n_cols+cols.number);
	query->GlobalModel.at(1) = arma::mat(arma::size(*global_learner->getModel().at(1)), arma::fill::zeros);
	query->GlobalModel.at(1) += *global_learner->getModel().at(1);
}

void coordinator::Progress(){
	// Query thr accuracy of the global model.
	if(Q->config.learning_algorithm == "ELM"){
		query->accuracy = Q->queryAccuracy(global_learner);
	}
	
	cout << "Global model of network " << net()->name() << "." << endl;
	cout << "accuracy : " << std::setprecision(6) << query->accuracy << "%" << endl;
	cout << "Number of rounds : " << num_rounds << endl;
	cout << "Number of subrounds : " << num_subrounds << endl;
	cout << "Total updates : " << total_updates << endl;
	cout << endl;
}

double coordinator::getAccuracy(){
	query->accuracy = Q->queryAccuracy(global_learner);
	return query->accuracy;
}

vector<size_t>  coordinator::Statistics() const{
	vector<size_t> stats;
	stats.push_back(num_rounds);
	stats.push_back(num_subrounds);
	stats.push_back(sz_sent);
	stats.push_back(0);
	return stats;
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
	|| Q->config.learning_algorithm == "MLP"
	|| Q->config.learning_algorithm == "ELM"){
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
	total_updates+=batch.n_cols;
	if(query->GlobalModel.size()==0){
		vector<arma::SizeMat> model_sizes = global_learner->modelDimensions();
		query->initializeGlobalModel(model_sizes);
		for(arma::SizeMat sz:model_sizes)
			Mean.push_back(arma::mat(sz, arma::fill::zeros));
	}
}

void coordinator::end_warmup(){
	query->update_estimate_v2(global_learner->getModel());
	start_round();
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
	}else if (cfg().learning_algorithm == "ELM"){
		global_learner = new ELM_Classifier(cfg().cfgfile, cfg().network_name);
	}else if (cfg().learning_algorithm == "MLP"){
		global_learner = new MLP_Classifier(cfg().cfgfile, cfg().network_name);
	}else if(cfg().learning_algorithm == "PA_Reg"){
		global_learner = new PassiveAgressiveRegression(cfg().cfgfile, cfg().network_name);
//	}else if(cfg().learning_algorithm == "NN_Reg"){
//		global_learner = new NN_Regressor(cfg().cfgfile, cfg().network_name);
	}
	if(cfg().learning_algorithm != "ELM")
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


/*********************************************
	node
*********************************************/

oneway learning_node::reset(const safezone& newsz){
	szone = newsz;       // Reset the safezone object
	datapoints_seen = 0; // Reset the drift vector
	_learner->update_model(szone.getSZone()->getGlobalModel()); // Updates the parameters of the local learner
}

model_state learning_node::get_drift(){
	// Getting the drift vector is done as getting the local statistic
	szone(drift, _learner->getModel(), 1.);
	return model_state(drift, _learner->getNumOfUpdates());
}

void learning_node::set_drift(model_state mdl){
	// Update the local learner with the model sent by the coordinator
	_learner->update_model(mdl._model);
}

void learning_node::update_stream(arma::mat& batch, arma::mat& labels){
	
	if (cfg().learning_algorithm == "ELM"){
		
		size_t alpha_rows = batch.n_rows - _learner->getHModel().at(0)->n_rows;
		size_t beta_cols = labels.n_rows - _learner->getModel().at(1)->n_cols;
		
		if(alpha_rows>0 && beta_cols>0){
			arma::mat newA = arma::zeros<arma::mat>(_learner->getHModel().at(0)->n_rows+alpha_rows, _learner->getHModel().at(0)->n_cols);
			
			newA.rows(0, _learner->getHModel().at(0)->n_rows-1) += *_learner->getHModel().at(0); 
			newA.rows(_learner->getHModel().at(0)->n_rows, newA.n_rows-1) += 
			coord.hybrid_drift(this, int_num(alpha_rows), int_num(beta_cols)).sub_params;
			
			*_learner->getHModel().at(0) = newA;
		}else if(alpha_rows>0){
			arma::mat newA = arma::zeros<arma::mat>(_learner->getHModel().at(0)->n_rows+alpha_rows, _learner->getHModel().at(0)->n_cols);
			
			newA.rows(0, _learner->getHModel().at(0)->n_rows-1) += *_learner->getHModel().at(0); 
			newA.rows(_learner->getHModel().at(0)->n_rows, newA.n_rows-1) += 
			coord.virtual_drift(this, int_num(alpha_rows)).sub_params;
			
			*_learner->getHModel().at(0) = newA;
		}else if(beta_cols>0){
			coord.real_drift(this, int_num(beta_cols));
		}	
	}
	
	_learner->fit(batch,labels);
	datapoints_seen += batch.n_cols;
	
	if( ml_safezone_function* entity = dynamic_cast<Variance_safezone_func*>(szone.getSZone()) ){
		if(szone(datapoints_seen) <= 0){
			datapoints_seen = 0;
			szone(drift, _learner->getModel(), 1.);
			if(szone.getSZone()->checkIfAdmissible_v2(drift)<0.)
				coord.local_violation(this);
		}
	}else{
		if(szone(datapoints_seen) <= 0.)
			coord.local_violation(this);
	}
}

oneway learning_node::set_HStatic_variables(const p_model_state& SHParams){
	_learner->restoreModel(SHParams._model);
}

oneway learning_node::augmentAlphaMatrix(matrix_message params){
	arma::mat newA = arma::zeros<arma::mat>(_learner->getHModel().at(0)->n_rows+params.sub_params.n_rows, 
	                                        _learner->getHModel().at(0)->n_cols);
	newA.rows(0, _learner->getHModel().at(0)->n_rows-1) += *_learner->getHModel().at(0); 
	newA.rows(_learner->getHModel().at(0)->n_rows, newA.n_rows-1) += params.sub_params;
	*_learner->getHModel().at(0) = newA;
}

oneway learning_node::augmentBetaMatrix(int_num cols){
	_learner->handleRD(cols.number);
}


void learning_node::initializeLearner(){
	if (cfg().learning_algorithm == "PA"){
		_learner = new PassiveAgressiveClassifier(cfg().cfgfile, cfg().network_name);
	}else if (cfg().learning_algorithm == "ELM"){
		_learner = new ELM_Classifier(cfg().cfgfile, cfg().network_name);
	}else if(cfg().learning_algorithm == "MLP"){
		_learner = new MLP_Classifier(cfg().cfgfile, cfg().network_name);
	}else if(cfg().learning_algorithm == "PA_Reg"){
		_learner = new PassiveAgressiveRegression(cfg().cfgfile, cfg().network_name);
//	}else if(cfg().learning_algorithm == "NN_Reg"){
//		_learner = new NN_Regressor(cfg().cfgfile, cfg().network_name);
	}
	if(cfg().learning_algorithm != "ELM")
		_learner->initializeModel(Q->testSet->n_rows);
	
}

void learning_node::setup_connections(){
	num_sites = coord.proc()->k;
}


ML_gm_networks::GM_Net::GM_Net(const set<source_id>& _hids, const string& _name, continuous_query* _Q)
: gm_learning_network_t(_hids, _name, _Q){
	this->set_protocol_name("ML_GM");
}