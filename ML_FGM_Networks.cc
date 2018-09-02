#include "ML_FGM_Networks.hh"

using namespace ml_gm_proto;
using namespace ml_gm_proto::ML_fgm_networks;


/*********************************************
	coordinator
*********************************************/

void coordinator::start_round(){
	
	// Resets.
	cnt = 0;
	counter = 0;
	for(size_t i=0; i<Beta.size(); i++){
		Beta.at(i).zeros();
	}
	
	// Calculating the new phi, quantum and the minimum acceptable value for phi.
	phi = k*safe_zone->Zeta(query->GlobalModel);
	quantum = phi/(2*k);
	assert(quantum>0);
	barrier = cfg().precision*phi;
	
	// Send new safezone.
	for(auto n : net()->sites) {
		if (cfg().learning_algorithm=="ELM" && num_rounds==0){
			proxy[n].set_HStatic_variables(p_model_state(global_learner->getHModel(), 0));
		}
		sz_sent ++;
		proxy[n].reset(safezone(safe_zone), float_value(quantum));
		node_bool_drift[n] = 0;
	}
	
	num_rounds++;
	num_subrounds++;
}

oneway coordinator::send_increment(increment inc){
	counter += inc.increase;
	if(counter>k){
		phi = 0.;
		// Collect all data
		for(auto n : node_ptr) {
			phi += proxy[n].get_zed_value().value;
		}
		if(phi>=barrier){
			counter = 0;
			quantum = phi/(2*k);
			assert(quantum>0);
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

void coordinator::fetch_updates(node_t* node){
	model_state up = proxy[node].get_drift();
	if( !arma::approx_equal(arma::mat(arma::size(up._model.at(0)),arma::fill::zeros), up._model.at(0), "absdiff", 1e-6) ){
		if(node_bool_drift[node]==0){
			node_bool_drift[node] = 1;
			cnt++;
		}
		for(size_t i=0; i<up._model.size(); i++){
			Params.at(i) += up._model.at(i);
		}
	}
	total_updates += up.updates;
}

// Rebalancing attempt.
void coordinator::Rebalance() {
	
	for(size_t i=0;i<Beta.size();i++){
		Beta.at(i) += Params.at(i);
	}
	for(size_t i=0;i<Beta.size();i++){
		Params.at(i) = query->GlobalModel.at(i) + (2./cnt)*Beta.at(i);
	}
	
	//phi = (k/2)*safe_zone->Zeta(Params) + (k/2)*safe_zone->Zeta(query->GlobalModel);
	phi = (cnt/2)*safe_zone->Zeta(Params) + ( (cnt/2)+(k-cnt) )*safe_zone->Zeta(query->GlobalModel);
	
	for(size_t i=0;i<Params.size();i++){
		Params.at(i).zeros();
	}
	
	if (phi>=1.*barrier){
		counter = 0;
		quantum = phi/(2*k);
		assert(quantum>0);
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
void coordinator::finish_round() {

	for(size_t i=0; i<Beta.size(); i++)
		Beta.at(i) *= std::pow(cnt,-1);

	// New round
	query->update_estimate_v2(Beta); 
	global_learner->update_model(query->GlobalModel);
	
	start_round();

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
	Beta.at(1).resize(Beta.at(1).n_rows, Beta.at(1).n_cols+cols.number);
	Params.at(1).resize(Params.at(1).n_rows, Params.at(1).n_cols+cols.number);
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
	Beta.at(1).resize(Beta.at(1).n_rows, Beta.at(1).n_cols+cols.number);
	Params.at(1).resize(Params.at(1).n_rows, Params.at(1).n_cols+cols.number);
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
	cout << "Number of rebalances : " << num_rebalances << endl;
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
	stats.push_back(num_rebalances);
	stats.push_back(sz_sent);
	return stats;
}

void coordinator::warmup(arma::mat& batch, arma::mat& labels){
	global_learner->fit(batch,labels);
	total_updates+=batch.n_cols;
	if(query->GlobalModel.size()==0){
		vector<arma::SizeMat> model_sizes = global_learner->modelDimensions();
		query->initializeGlobalModel(model_sizes);
		for(arma::SizeMat sz : model_sizes){
			Beta.push_back(arma::mat(sz, arma::fill::zeros));
			Params.push_back(arma::mat(sz, arma::fill::zeros));
		}
	}
}

void coordinator::end_warmup(){
	query->update_estimate_v2(global_learner->getModel());
	start_round();
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

void coordinator::setup_connections(){
	using boost::adaptors::map_values;
	proxy.add_sites(net()->sites);
	for(auto n : net()->sites) {
		node_index[n] = node_ptr.size();
		node_bool_drift[n] = 0;
		node_ptr.push_back(n);
	}
	k = node_ptr.size();
}

coordinator::coordinator(network_t* nw, continuous_query* _Q)
: 	process(nw), proxy(this),
	Q(_Q),
	k(0),
	num_rounds(0), num_subrounds(0),
	sz_sent(0), total_updates(0), num_rebalances(0){
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

oneway learning_node::reset(const safezone& newsz, const float_value qntm){
	counter = 0;
	szone = newsz;                                                    // Reset the safezone object
	quantum = 1.*qntm.value;                                          // Reset the quantum
	_learner->update_model(szone.getSZone()->getGlobalModel());       // Updates the parameters of the local learner
	zeta = szone.getSZone()->Zeta(_learner->getModel());              // Reset zeta
	
	// Initializng the helping matrices if they are not yet initialized.
	if(E_Delta.empty()){
		Delta_Vector.clear();
		E_Delta.clear();
		for(size_t i=0; i<_learner->getModel().size(); i++){
			arma::mat tmp1 = arma::mat(arma::size(*_learner->getModel().at(i)), arma::fill::zeros);
			arma::mat tmp2 = arma::mat(arma::size(*_learner->getModel().at(i)), arma::fill::zeros);
			Delta_Vector.push_back(tmp1);
			E_Delta.push_back(tmp2);
		}
	}
	
	// Reseting the E_Delta vector.
	for (size_t i=0; i<_learner->getModel().size(); i++){
		E_Delta.at(i) = *_learner->getModel().at(i);
	}
}

oneway learning_node::take_quantum(const float_value qntm){
	counter = 0;                                            // Reset counter
	quantum = 1.*qntm.value;                                // Update the quantum
	zeta = szone(_learner->getModel(), E_Delta);            // Update zeta
}

oneway learning_node::rebalance(const float_value qntm){
	counter = 0;											// Reset counter
	quantum = 1.*qntm.value;                                // Update the quantum
	
	// Updating the E_Delta vector.
	for (size_t i=0; i<_learner->getModel().size(); i++){
		E_Delta.at(i) = *_learner->getModel().at(i);
	}
	
	zeta = szone(_learner->getModel(), E_Delta);            // Reset zeta
}

model_state learning_node::get_drift(){
	// Getting the delta vector is done as getting the local statistic.
	Delta_Vector.clear();
	for(size_t i=0; i<E_Delta.size(); i++){
		arma::mat dr = arma::mat(arma::size(*_learner->getModel().at(i)), arma::fill::zeros);
		dr = *_learner->getModel().at(i) - E_Delta.at(i);
		Delta_Vector.push_back(dr);
	}
	return model_state(Delta_Vector, _learner->getNumOfUpdates());
}

float_value learning_node::get_zed_value(){
	return float_value(szone(_learner->getModel(), E_Delta));
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
			E_Delta.at(1).resize(_learner->getModel().at(1)->n_rows,_learner->getModel().at(1)->n_cols+beta_cols);
			Delta_Vector.at(1).resize(_learner->getModel().at(1)->n_rows,_learner->getModel().at(1)->n_cols+beta_cols);
		}else if(alpha_rows>0){
			arma::mat newA = arma::zeros<arma::mat>(_learner->getHModel().at(0)->n_rows+alpha_rows, _learner->getHModel().at(0)->n_cols);
			
			newA.rows(0, _learner->getHModel().at(0)->n_rows-1) += *_learner->getHModel().at(0); 
			newA.rows(_learner->getHModel().at(0)->n_rows, newA.n_rows-1) += 
			coord.virtual_drift(this, int_num(alpha_rows)).sub_params;
			
			*_learner->getHModel().at(0) = newA;
		}else if(beta_cols>0){
			coord.real_drift(this, int_num(beta_cols));
			E_Delta.at(1).resize(_learner->getModel().at(1)->n_rows,_learner->getModel().at(1)->n_cols+beta_cols);
			Delta_Vector.at(1).resize(_learner->getModel().at(1)->n_rows,_learner->getModel().at(1)->n_cols+beta_cols);
		}	
	}
	
	_learner->fit(batch,labels);
	
	if( ml_safezone_function* entity = dynamic_cast<Variance_safezone_func*>(szone.getSZone()) ){
		int c_now = std::floor((zeta-szone(_learner->getModel(), E_Delta))/quantum);
		if(c_now-counter>0){
			coord.send_increment(increment(c_now-counter));
			counter = c_now;
		}
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
	E_Delta.at(1).resize(_learner->getModel().at(1)->n_rows,_learner->getModel().at(1)->n_cols);
	Delta_Vector.at(1).resize(_learner->getModel().at(1)->n_rows,_learner->getModel().at(1)->n_cols);
}

void learning_node::initializeLearner(){
	// Initializing the learner.
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
		
	cout << "Local site " << this->name() << " initialized its network." << endl;
}

void learning_node::setup_connections(){
	num_sites = coord.proc()->k;
}


ML_fgm_networks::FGM_Net::FGM_Net(const set<source_id>& _hids, const string& _name, continuous_query* _Q)
: gm_learning_network_t(_hids, _name, _Q){
	this->set_protocol_name("ML_FGM");
}