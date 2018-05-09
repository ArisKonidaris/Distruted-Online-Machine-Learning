#include "feeders.hh"

using namespace feeders;
using namespace feeders::MLPACK_feeders;

/*********************************************
	feeder
*********************************************/

feeder::feeder(string cfg):
config_file(cfg){
	Json::Value root;
	std::ifstream cfgfile(config_file); // Parse from JSON file.
	cfgfile >> root;
	
	long int random_seed = (long int)root["simulations"].get("seed",0).asInt64();
	seed = time(&random_seed);
	
	negative_labels = root["simulations"].get("negative_labels",true).asBool();
	if(negative_labels!=true && negative_labels!=false){
		cout << endl << "Incorrect negative labels." << endl;
		cout << "Acceptable negative labels parameters : true, false" << endl;
		throw;
	}
	
	string learning_problem = root["simulations"].get("learning_problem","NoProblem").asString();
	if(learning_problem!= "classification" && learning_problem!= "regression"){
		cout << endl << "Incorrect learning problem given." << endl;
		cout << "Acceptable learning problems are : 'classification', 'regression'" << endl;
		throw;
	}else{
		if(learning_problem == "regression")
			negative_labels = false;
	}
}

feeder::~feeder() { }

void feeder::initializeSimulation(){
	
	cout << "Initializing the star nets..." << endl << endl << endl;
	Json::Value root;
	std::ifstream cfgfile(config_file); // Parse from JSON file.
	cfgfile >> root;
	
	source_id count = 0;
	size_t number_of_networks = root["simulations"].get("number_of_networks",0).asInt();
	for(size_t i = 1; i <= number_of_networks; i++){
		
		string net_name = root["simulations"].get("net_name_"+std::to_string(i), "NoNet").asString();
		string learning_algorithm = root["gm_network_"+net_name].get("learning_algorithm", "NoAlgo").asString();
		
		cout << "Initializing the network " << net_name << " with " << learning_algorithm <<  " learner." << endl;
		
		if(learning_algorithm == "PA" || learning_algorithm == "Kernel_PA" || learning_algorithm == "MLP"){
			auto query = new Classification_query(&testSet, &testResponses, config_file, net_name);
			addQuery(query);
			cout << "Query added." << endl;
		}else if(learning_algorithm == "PA_Reg" || learning_algorithm == "NN_Reg" ){
			auto query = new Regression_query(&testSet, &testResponses, config_file, net_name);
			addQuery(query);
			cout << "Query added." << endl;
		}
		
		source_id number_of_nodes = (source_id)root["gm_network_"+net_name].get("number_of_local_nodes", 1).asInt64();
		set<source_id> node_ids;
		for(source_id j = 1; j <= number_of_nodes; j++){
			node_ids.insert(count+j);
		}
		count += number_of_nodes+1; // We add one because of the coordinator.
		
		auto net = new network(node_ids, net_name, _query_container.at(i-1));
		addNet(net);
		cout << "Net " << net_name << " initialized." << endl << endl;
	}
	cout << endl << "Netwoks initialized." << endl;
}

void feeder::printStarNets() const {
	cout << endl << "Printing the nets." << endl;
	cout << "Number of networks : " << _net_container.size() << endl;
	for(auto net:_net_container){
		cout << endl;
		cout << "Network Name : " << net->name() << endl;
		cout << "Number of nodes : " << net->sites.size() << endl;
		cout << "Coordinator " << net->hub->name() << " with address " << net->hub->addr() << endl;
		for(size_t j = 0; j < net->sites.size(); j++){
			cout << "Site " << net->sites.at(j)->name()  << " with address " << net->sites.at(j)->site_id() << endl;
		}
	}
}


/*********************************************
	Random_Feeder
*********************************************/

Random_Feeder::Random_Feeder(string cfg)
:feeder(cfg){
	try{
		Json::Value root;
		std::ifstream cfgfile(config_file); // Parse from JSON file.
		cfgfile >> root;
		
		linearly_seperable = root["tests_Generated_Data"].get("linearly_seperable",true).asBool();
		if(linearly_seperable!=true && linearly_seperable!=false){
			cout << endl << "Incorrect parameter linearly_seperable." << endl;
			cout << "The linearly_seperable parameter must be a boolean." << endl;
			throw;
		}
		
		batchSize = root["tests_Generated_Data"].get("batch_size",1).asInt64();
		warmupSize = root["tests_Generated_Data"].get("warmup_size",500).asInt64();
			
		number_of_features = root["tests_Generated_Data"].get("number_of_features",20).asInt64();
		if(number_of_features <= 0){
			cout << endl << "Incorrect parameter number_of_features" << endl;
			cout << "Acceptable number_of_features parameters are all the positive integers." << endl;
			throw;
		}
			
		test_size = root["tests_Generated_Data"].get("test_size",100000).asDouble();
		if(test_size<0){
			cout << endl << "Incorrect parameter test_size." << endl;
			cout << "Acceptable test_size parameters are all the positive integers." << endl;
			throw;
		}
		
		std::srand (seed);
		
		cout << "test_size : " << test_size << endl << endl;
		
		// Create the test dataset.
		makeTestDataset();
		targets = 1;
		
	}catch(...){
		cout << endl << "Something went wrong in Random_Feeder object construction." << endl;
		throw;
	}
}

void Random_Feeder::GenNewTarget(){
	target = arma::zeros<arma::mat>(number_of_features,1);
	for(size_t i = 0; i < target.n_elem;i++){
		if( (double)std::rand()/RAND_MAX <= std::sqrt(1-std::pow(2,-(1/number_of_features))) ){
			target(i,0) = 1.;
		}
	}
	cout << "New Target" << endl;
	targets++;
}

void Random_Feeder::makeTestDataset(){
	GenNewTarget();
	testSet = arma::zeros<arma::mat>(number_of_features,test_size);
	testResponses = arma::zeros<arma::mat>(1,test_size);
	
	for(size_t i = 0; i < test_size; i++){
		arma::dvec point = GenPoint();
		testSet.col(i) = point;
		if(arma::dot(point,target.unsafe_col(0))>=1.){
			testResponses(0,i) = 1.;
		}else{
			if(negative_labels){
				testResponses(0,i) = -1.;
			}
		}
	}
	
}

arma::dvec Random_Feeder::GenPoint(){
	arma::dvec point = arma::zeros<arma::dvec>(number_of_features);
	for(size_t j = 0; j < number_of_features; j++){
		if( (double)std::rand()/RAND_MAX <= std::sqrt(1-std::pow(2,-(1/number_of_features))) ){
			point(j) = 1.;
		}
	}
	return point;
}

void Random_Feeder::TrainNetworks(){
	size_t count = 0; // Count the number of processed elements.
	
	bool warm = false; // Variable indicating if the networks are warmed up.
	size_t degrees = 0; // Number of warmup datapoints read so far by the networks.
	
	//numOfMaxRounds// = 100000; 
	while(count<numOfPoints){
		
		if( (double)std::rand()/RAND_MAX <= 0.0001 ){
			makeTestDataset();
		}
		
		arma::mat point = arma::zeros<arma::mat>(number_of_features,1);
		arma::mat label = arma::zeros<arma::mat>(1,1);
		point.col(0) = GenPoint();
		
		if(arma::dot(point.unsafe_col(0), target.unsafe_col(0))>=1.){
			label(0,0) = 1.;
		}else{
			if(negative_labels){
				label(0,0) = -1.;
			}
		}
							   
		// Update the number of processed elements.
		count += 1;
			
		if(!warm){ // We warm up the networks.
			degrees += 1;
			for(size_t i = 0; i < _net_container.size(); i++){
				if(_net_container.at(i)->Q->config.learning_algorithm == "MLP"){
					label.transform( [](double val) { return (val == -1.) ? 1.: val+1.; } );
					_net_container.at(i)->warmup(point,label);
					label.transform( [](double val) { return (val == 1.) ? -1.: val-1; } );
				}else{
					_net_container.at(i)->warmup(point,label);
				}
			}
			if(degrees==warmupSize){
				warm = true;
				for(size_t i = 0; i < _net_container.size(); i++){
					_net_container.at(i)->start_round(); // Each hub initializes the first round.
				}
			}
			continue;
		}
		
		Train(point,label);
		
		if(count%5000 == 0){
			cout << "count : " << count << endl;
		}
		
	}
	
	for(auto net:_net_container){
		net->process_fini();
	}
	count = 0;
	cout << "Targets : " << targets << endl;
}

void Random_Feeder::Train(arma::mat& point, arma::mat& label){
	for(auto net:_net_container){
		size_t random_node = std::rand()%(net->sites.size());
		if(net->cfg().learning_algorithm == "MLP"){
			
			/// Transform the label to the mlpack's Neural Net format.
			/// After training the transformation is reversed. 
			label.transform( [](double val) { return (val == -1.) ? 1.: val+1.; } );
			net->process_record(random_node, point, label);
			label.transform( [](double val) { return (val == 1.) ? -1.: val-1; } );
			
		}else{
			/// Train on data point.
			net->process_record(random_node, point, label);
		}
	}
}
