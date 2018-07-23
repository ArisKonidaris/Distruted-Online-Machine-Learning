#ifndef __FEEDERS_HH__
#define __FEEDERS_HH__

#include "ML_GM_Networks.hh"
#include "ML_FGM_Networks.hh"
#include "DL_GM_Networks.hh"
#include "DL_FGM_Networks.hh"
#include <fstream>
#include <iostream>

namespace feeders{

using namespace ml_gm_proto;

namespace MLPACK_feeders{

using namespace ml_gm_proto::ML_gm_networks;
using namespace ml_gm_proto::ML_fgm_networks;
	
/*	
	A vector container for the networks.
	*/
template<typename distrNetType>
class net_container : public vector<distrNetType*>
{
public:
	using vector<distrNetType*>::vector;

	inline void join(distrNetType* net) { this->push_back(net); }
	inline void leave(int const i) { this->erase(this->begin()+i); }

};

/*	
	A vector container for the queries.
	*/
class query_container : public vector<continuous_query*>
{
	public:
	using vector<continuous_query*>::vector;

	inline void join(continuous_query* qry) { this->push_back(qry); }
	inline void leave(int const i) { this->erase(this->begin()+i); }
	
};

/* 
	A feeders purpose is to synchronize the testing of the networks
	by providing the appropriate data stream to the nodes of each net.
	*/
template<typename distrNetType>
class feeder{
protected:
	string config_file; // JSON file to read the hyperparameters.
	time_t seed; // The seed for the random generator.
	size_t batchSize; // The batch learning size.
	size_t warmupSize; // The size of the warmup dataset.
	arma::mat testSet; // Test dataset for validation of the classification algorithm.
	arma::mat testResponses; // Arma row vector containing the labels of the evaluation set.
	
	size_t test_size; // Size of test dataset. [optional]
	bool negative_labels; // If true it sets 0 labels to -1. [optional]
	
	net_container<distrNetType> _net_container; // A container for networks.
	query_container _query_container; // A container for queries.
	
	// Stream Distribution
	bool uniform_distr;
	vector<vector<set<size_t>>> net_dists;
	float B_prob;
	float site_ratio;
	
	// Statistics collection.
	vector<chan_frame> stats;
	vector<vector<vector<size_t>>> differential_communication;
	size_t msgs;
	size_t bts;
	vector<vector<double>> differential_accuracy;
	
public:
	/** 	
		Constructor, destructor.
		**/
	feeder(string cfg);
	virtual ~feeder();
	
	/** 	
	    Method that creates the test dataset.
		This method passes one time through the entire dataset,
		if the dataset is stored in a hdf5 file.
		**/
	virtual void makeTestDataset() { }
	
	/* Method that puts a network in the network container. */
	void addQuery(continuous_query* qry) { _query_container.join(qry); }
	
	/* Method that puts a network in the network container. */
	void addNet(distrNetType* net) { _net_container.join(net); }
	
	/* Method initializing all the networks. */
	void initializeSimulation();
	
	/* Method that prints the star learning network for debbuging purposes. */
	void printStarNets() const;
	
	/* Method that gathers communication info after each streaming batch. */
	void gatherDifferentialInfo();
	
	// Getters.
	inline arma::mat& getTestSet() { return testSet; }
	inline arma::mat& getTestSetLabels() { return testResponses; }
	inline arma::mat* getPTestSet() { return &testSet; }
	inline arma::mat* getPTestSetLabels() { return &testResponses; }
	inline size_t getRandomInt(size_t maxValue) { return std::rand() % maxValue; }
	
	virtual inline size_t getNumberOfFeatures() { return 0; }
	virtual void getStatistics() { }
};

template<typename distrNetType>
feeder<distrNetType>::feeder(string cfg)
:config_file(cfg){
	Json::Value root;
	std::ifstream cfgfile(config_file); // Parse from JSON file.
	cfgfile >> root;
	
	long int random_seed = (long int)root["simulations"].get("seed",0).asInt64();
	if(random_seed>=0){
		seed = random_seed;
	}else{
		seed = time(&random_seed);
	}
	std::srand (seed);
	
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
	
	// Get the stream distribution.
	uniform_distr = root[root["simulations"].get("stream_distribution", "No_Distribution").asString()]
					.get("uniform", true).asBool();
	if(!uniform_distr){
		B_prob = root[root["simulations"].get("stream_distribution", "No_Distribution").asString()]
				 .get("B_prob",-1.).asFloat();
		if(B_prob>1. || B_prob<0.){
			cout << "Invalid parameter B_prob. Probabilities must be in the interval [0,1]" << endl;
			throw;
		}
		site_ratio = root[root["simulations"].get("stream_distribution", "No_Distribution").asString()]
			         .get("site_ratio",-1.).asFloat();
		 if(site_ratio>1. || site_ratio<0.){
			cout << "Invalid parameter site_ratio. Ratios must be in the interval [0,1]" << endl;
			throw;
		}
	}
	
}

template<typename distrNetType>
feeder<distrNetType>::~feeder() { }

template<typename distrNetType>
void feeder<distrNetType>::initializeSimulation(){
	
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
		
		if(learning_algorithm == "PA" || learning_algorithm == "ELM" || learning_algorithm == "MLP"){
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
		
		auto net = new distrNetType(node_ids, net_name, _query_container.at(i-1));
		addNet(net);
		cout << "Net " << net_name << " initialized." << endl << endl;
	}
	for(auto net:_net_container){
		// Initializing the differential communication statistics.
		stats.push_back(chan_frame(net));
		vector<vector<size_t>> dif_com;
		vector<size_t> dif_msgs;
		vector<size_t> dif_bts;
		dif_msgs.push_back(0);
		dif_bts.push_back(0);
		dif_com.push_back(dif_msgs);
		dif_com.push_back(dif_bts);
		differential_communication.push_back(dif_com);
		
		vector<double> dif_acc;
		dif_acc.push_back(0.);
		differential_accuracy.push_back(dif_acc);
		
		// Initializing the stream distributions of the sites of the network.
		if(!uniform_distr){
			set<size_t> B;
			set<size_t> B_compl;
			vector<set<size_t>> net_distr;
			for(size_t i=0; i<net->sites.size(); i++){
				B_compl.insert(i);
			}
			for(size_t i=0; i<std::floor(net->sites.size()*site_ratio); i++){
				size_t n = std::rand()%(net->sites.size());
				while(B.find(n) != B.end()){
					n = std::rand()%(net->sites.size());
				}
				B.insert(n);
				B_compl.erase(n);
			}
			net_distr.push_back(B);
			net_distr.push_back(B_compl);
			net_dists.push_back(net_distr);
		}
	}
	msgs = 0;
	bts = 0;
	cout << endl << "Networks initialized." << endl;
}

template<typename distrNetType>
void feeder<distrNetType>::printStarNets() const {
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

template<typename distrNetType>
void feeder<distrNetType>::gatherDifferentialInfo() {
	// Gathering the info of the communication triggered by the streaming batch.
	for(size_t i=0; i<_net_container.size(); i++){
		size_t batch_messages = 0;
		size_t batch_bytes = 0;
		
		for(auto chnl:stats.at(i)){
			batch_messages += chnl->messages_received();
			batch_bytes += chnl->bytes_received();
		}
		
		differential_communication.at(i).at(0)
								  .push_back( batch_messages - msgs );
		
		differential_communication.at(i).at(1)
								  .push_back( batch_bytes - bts  );
								  
		msgs = batch_messages;
		bts = batch_bytes;
		
		//differential_accuracy.at(i).push_back(_net_container.at(i)->hub->getAccuracy());
	}
}

template<typename distrNetType>
class Random_Feeder : public feeder<distrNetType>{
protected:
	size_t test_size; // Starting test data point.
	size_t number_of_features; // The number of features of each datapoint.
	bool linearly_seperable; // Determines if the random dataset is linearly seperable.
	arma::mat target; // The moving target disjunction of the stream.
	size_t targets; // The total number of changed targets.
	
	size_t numOfPoints = 12800000; // Total number of datapoints.
	size_t numOfMaxRounds = 100000; // Maximum number of monitored rounds.
	
public:
	/** 	
		Constructor.
		**/
	Random_Feeder(string cfg);
	
	/**	
	    A destructor for the class.
		Probably needs some fixing later on.
		**/
	~Random_Feeder() { }
	
	void makeTestDataset() override;
	
	void GenNewTarget();
	
	arma::dvec GenPoint();
	
	void TrainNetworks();
	
	void Train(arma::mat& batch, arma::mat& labels);
	
	void getStatistics() override { }
	
	inline size_t getNumberOfFeatures() override { return number_of_features; }
};

template<typename distrNetType>
Random_Feeder<distrNetType>::Random_Feeder(string cfg)
:feeder<distrNetType>(cfg){
	try{
		Json::Value root;
		std::ifstream cfgfile(this->config_file); // Parse from JSON file.
		cfgfile >> root;
		
		linearly_seperable = root["tests_Generated_Data"].get("linearly_seperable",true).asBool();
		if(linearly_seperable!=true && linearly_seperable!=false){
			cout << endl << "Incorrect parameter linearly_seperable." << endl;
			cout << "The linearly_seperable parameter must be a boolean." << endl;
			throw;
		}
		
		this->batchSize = root["tests_Generated_Data"].get("batch_size",1).asInt64();
		this->warmupSize = root["tests_Generated_Data"].get("warmup_size",500).asInt64();
			
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
		
		std::srand (this->seed);
		
		cout << "test_size : " << test_size << endl << endl;
		
		// Create the test dataset.
		makeTestDataset();
		targets = 1;
		
	}catch(...){
		cout << endl << "Something went wrong in Random_Feeder object construction." << endl;
		throw;
	}
}

template<typename distrNetType>
void Random_Feeder<distrNetType>::GenNewTarget(){
	target = arma::zeros<arma::mat>(number_of_features,1);
	for(size_t i = 0; i < target.n_elem;i++){
		if( (double)std::rand()/RAND_MAX <= std::sqrt(1-std::pow(2,-(1/number_of_features))) ){
			target(i,0) = 1.;
		}
	}
	cout << "New Target" << endl;
	targets++;
}

template<typename distrNetType>
void Random_Feeder<distrNetType>::makeTestDataset(){
	GenNewTarget();
	this->testSet = arma::zeros<arma::mat>(number_of_features, this->test_size);
	this->testResponses = arma::zeros<arma::mat>(1, this->test_size);
	
	for(size_t i = 0; i < test_size; i++){
		arma::dvec point = GenPoint();
		this->testSet.col(i) = point;
		if(arma::dot(point,target.unsafe_col(0))>=1.){
			this->testResponses(0,i) = 1.;
		}else{
			if(this->negative_labels){
				this->testResponses(0,i) = -1.;
			}
		}
	}
}

template<typename distrNetType>
arma::dvec Random_Feeder<distrNetType>::GenPoint(){
	arma::dvec point = arma::zeros<arma::dvec>(number_of_features);
	for(size_t j = 0; j < number_of_features; j++){
		if( (double)std::rand()/RAND_MAX <= std::sqrt(1-std::pow(2,-(1/number_of_features))) ){
			point(j) = 1.;
		}
	}
	return point;
}

template<typename distrNetType>
void Random_Feeder<distrNetType>::TrainNetworks(){
	size_t count = 0; // Count the number of processed elements.
	
	bool warm = false; // Variable indicating if the networks are warmed up.
	size_t degrees = 0; // Number of warmup datapoints read so far by the networks.
	
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
			if(this->negative_labels){
				label(0,0) = -1.;
			}
		}
							   
		// Update the number of processed elements.
		count += 1;
			
		if(!warm){ // We warm up the networks.
			degrees += 1;
			for(size_t i = 0; i < this->_net_container.size(); i++){
				if(this->_net_container.at(i)->Q->config.learning_algorithm == "MLP"){
					label.transform( [](double val) { return (val == -1.) ? 1.: val+1.; } );
					this->_net_container.at(i)->warmup(point,label);
					label.transform( [](double val) { return (val == 1.) ? -1.: val-1; } );
				}else{
					this->_net_container.at(i)->warmup(point,label);
				}
			}
			if(degrees==this->warmupSize){
				warm = true;
				for(size_t i = 0; i < this->_net_container.size(); i++){
					this->_net_container.at(i)->start_round(); // Each hub initializes the first round.
				}
			}
			continue;
		}
		
		Train(point,label);
		
		if(count%5000 == 0){
			cout << "count : " << count << endl;
		}
		
	}
	
	for(auto net:this->_net_container){
		net->process_fini();
	}
	count = 0;
	cout << "Targets : " << targets << endl;
}

template<typename distrNetType>
void Random_Feeder<distrNetType>::Train(arma::mat& point, arma::mat& label){
	for(auto net:this->_net_container){
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

template<typename distrNetType>
class HDF5_Feeder : public feeder<distrNetType>{
	typedef boost::shared_ptr<hdf5Source<double>> PointToSource;
protected:
	size_t start_test; // Starting test data point.
	
	PointToSource DataSource; // Data Source to read the dataset in streaming form.
public:
	/** 	
		Constructor.
		**/
	HDF5_Feeder(string cfg);
	
	/**	
	    A destructor for the class.
		Probably needs some fixing later on.
		**/
	~HDF5_Feeder();
	
	void makeTestDataset() override;
	
	void TrainNetworks();
	
	void Train(arma::mat& batch, arma::mat& labels);
	
	void getStatistics() override { }
	
	inline size_t getNumberOfFeatures() override { return DataSource->getDataSize()-1; }
	
};

template<typename distrNetType>
HDF5_Feeder<distrNetType>::HDF5_Feeder(string cfg)
:feeder<distrNetType>(cfg){
	try{
		Json::Value root;
		std::ifstream cfgfile(this->config_file); // Parse from JSON file.
		cfgfile >> root;
		
		// Test if the learning algorithms are compatible with this feeder.
		size_t number_of_networks = root["simulations"].get("number_of_networks",0).asInt();
		for(size_t i = 1; i <= number_of_networks; i++){
			string learning_algorithm = root["gm_network_"+root["simulations"].get("net_name_"+std::to_string(i), "NoNet").asString()]
			                            .get("learning_algorithm", "NoAlgo").asString();
			if(learning_algorithm=="ELM"){
				cout << "The Extreme Learning Classifier is not compatible with the HDF5_Feeder class." << endl;
				throw;
			}
		}
		
		this->batchSize = root["tests_hdf5_Data"].get("batch_size",1).asInt64();
		this->warmupSize = root["tests_hdf5_Data"].get("warmup_size",500).asInt64();
		
		// Initialize data source.
		DataSource = getPSource<double>(root["data"].get("file_name","No file name given").asString(),
									    root["data"].get("dataset_name","No file dataset name given").asString(),
									    false);
			
		double test_sz = root["tests_hdf5_Data"].get("test_size",-1).asDouble();
		if(test_sz<0.0 || test_sz>1.0){
			cout << endl << "Incorrect parameter test_size." << endl;
			cout << "Acceptable test_size parameters : [double] in range:[0.0, 1.0]" << endl;
			throw;
		}
						
		// Determining the size of the test Dataset.
		this->test_size = std::floor(this->test_sz*DataSource->getDatasetLength());
		std::srand (this->seed);
		start_test = std::rand()%(DataSource->getDatasetLength() - this->test_size+1);
		
		// Initializing the test dataset and its labels with zeros.
		this->testSet = arma::zeros<arma::mat>(DataSource->getDataSize(), this->test_size);
		this->testResponses = arma::zeros<arma::mat>(1, this->test_size);
		
		cout << endl << "start_test : " << start_test << endl;
		cout << "test_size : " << this->test_size << endl << endl;
		
		// Create the test dataset.
		makeTestDataset();
		
	}catch(...){
		cout << endl << "Something went wrong in HDF5_Feeder object construction." << endl;
		throw;
	}
}

template<typename distrNetType>
HDF5_Feeder< distrNetType >::~HDF5_Feeder(){
	DataSource = nullptr;
}

template<typename distrNetType>
void HDF5_Feeder< distrNetType >::makeTestDataset(){
	vector<double>& buffer = DataSource->getbuffer(); // Initialize the dataset buffer.
	size_t count = 0; // Count the number of processed elements.
	
	size_t tests_points_read = 0; // Test points read so far.
	size_t start = 0; // Position pointer to be user by the algorithm.
	while(DataSource->isValid()){
	
		arma::mat batch; // Arma column wise matrix containing the data points.
		
		// Load the batch from the buffer while removing the ids.
		batch = arma::mat(&buffer[0],
						  DataSource->getDataSize(),
						  (int)DataSource->getBufferSize(),
						  false,
						  false);
							   
		// Update the number of processed elements.
		count += DataSource->getBufferSize();
		
		if(count>=start_test && tests_points_read!=this->test_size){
			
			if(tests_points_read==0){
				start = start_test - (count-DataSource->getBufferSize()+1);
			}
			
			if(count < start_test + this->test_size -1){
				
				this->testSet.cols(tests_points_read, tests_points_read+DataSource->getBufferSize()-1-start) = 
				batch.cols(start, batch.n_cols-1); // Create the test dataset.
							 
				this->testResponses.cols(tests_points_read, tests_points_read+DataSource->getBufferSize()-1-start) =
				arma::conv_to<arma::mat>::from(this->testSet.cols(tests_points_read, tests_points_read+DataSource->getBufferSize()-1-start)
													        .row(batch.n_rows-1)); // Get the labels.
				
				batch.shed_cols(start, batch.n_cols-1); // Remove the test dataset from the batch.
				
				tests_points_read += DataSource->getBufferSize() - start;
				start = 0;
				
			}else{
				
				this->testSet.cols(tests_points_read, this->test_size-1) = 
				batch.cols(start, start+(this->test_size-tests_points_read)-1); // Create the test dataset.
															
				this->testResponses.cols(tests_points_read, this->test_size-1) = 
				arma::conv_to<arma::mat>::from(this->testSet.cols(tests_points_read, this->test_size-1)
													        .row(batch.n_rows-1)); // Get the labels.
				
				batch.shed_cols(start, start+(this->test_size-tests_points_read)-1); // Remove the test dataset from the batch.
				
				tests_points_read = this->test_size;
				
			}
			
			if(tests_points_read == this->test_size){
				this->testSet.shed_row(this->testSet.n_rows-1);
				// Replace every -1 label with 1 if necessary.
				if(this->negative_labels)
					this->testResponses.transform( [](double val) { return (val == 0.) ? -1.: val; } );
				break;
			}
			
		}
		
		// Get the next 1000 data points from disk to stream them.
		DataSource->advance();
	}
	DataSource->rewind();
}

template<typename distrNetType>
void HDF5_Feeder< distrNetType >::TrainNetworks(){
	vector<double>& buffer = DataSource->getbuffer(); // Initialize the dataset buffer.
	size_t count = 0; // Count the number of processed elements.
	
	bool warm = false; // Variable indicating if the networks are warmed up.
	size_t degrees = 0; // Number of warmup datapoints read so far by the networks.
	
	size_t tests_points_read = 0;
	size_t start = 0;
	while(DataSource->isValid()){
	
		arma::mat batch; // Arma column wise matrix containing the data points.
		
		// Load the batch from the buffer while removing the ids.
		batch = arma::mat(&buffer[0],
						  DataSource->getDataSize(),
						  (int)DataSource->getBufferSize(),
						  false,
						  false);
							   
		// Update the number of processed elements.
		count += DataSource->getBufferSize();
		
		// Discard the test set from the buffer.
		if(count>=start_test && tests_points_read!=this->test_size){
			if(tests_points_read==0){
				start = start_test - (count-DataSource->getBufferSize()+1);
			}
			if(count < start_test + this->test_size -1){
				batch.shed_cols(start, batch.n_cols-1); // Remove the test dataset from the batch.
				tests_points_read += DataSource->getBufferSize() - start;
				start = 0;
			}else{
				batch.shed_cols(start, start+(this->test_size-tests_points_read)-1); // Remove the test dataset from the batch.
				tests_points_read = this->test_size;
			}
		}
		
		// Train the nets
		if(batch.n_cols!=0){
			
			arma::mat labels = arma::conv_to<arma::mat>::from(batch.row(batch.n_rows-1)); // Get the labels.
			batch.shed_row(batch.n_rows-1); // Remove the labels.
			if(this->negative_labels){ // Replace every -1 label with 1 if necessary.
				labels.transform( [](double val) { return (val == 0.) ? -1.: val; } );
			}
			
			if(!warm){ // We warm up the networks.
				if(batch.n_cols <= this->warmupSize-degrees){ // A part of the warmup dataset is on the buffer.
					degrees += batch.n_cols;
					for(size_t i = 0; i < this->_net_container.size(); i++){
						if(this->_net_container.at(i)->Q->config.learning_algorithm == "MLP"){
							labels.transform( [](double val) { return (val == -1.) ? 1.: val+1.; } );
							this->_net_container.at(i)->warmup(batch,labels);
							labels.transform( [](double val) { return (val == 1.) ? -1.: val-1; } );
						}else{
							this->_net_container.at(i)->warmup(batch,labels);
						}
					}
					DataSource->advance();
					continue;
				}else{ // The last chunk of the warmup dataset.
					arma::mat stream_points = arma::mat(&batch.unsafe_col(0)(0) , batch.n_rows, this->warmupSize-degrees, false);
					arma::mat stream_labels = arma::mat(&labels.unsafe_col(0)(0), 1, this->warmupSize-degrees, false);
					
					warm = true;
					for(size_t i = 0; i < this->_net_container.size(); i++){
						if(this->_net_container.at(i)->Q->config.learning_algorithm == "MLP"){
							stream_labels.transform( [](double val) { return (val == -1.) ? 1.: val+1.; } );
							this->_net_container.at(i)->warmup(stream_points, stream_labels);
							stream_labels.transform( [](double val) { return (val == 1.) ? -1.: val-1; } );
						}else{
							this->_net_container.at(i)->warmup(stream_points, stream_labels);
						}
						this->_net_container.at(i)->start_round(); // Each hub initializes the first round.
					}
					
					// Remove the warmup dataset from the buffer.
					batch.shed_cols(0,this->warmupSize-degrees-1);
					labels.shed_cols(0,this->warmupSize-degrees-1);
				}
			}
			
			if(batch.n_cols!=0){ // The networks are warmed up so we train.
				// Do batch learning. The batch can be an integer in the interval [1:5000].
				size_t mod = batch.n_cols % this->batchSize;
				size_t num_of_batches = std::floor( batch.n_cols / this->batchSize );
				
				if( num_of_batches > 0 ){
					for(unsigned i = 0; i < num_of_batches; ++i){
						arma::mat stream_points = arma::mat(&batch.unsafe_col(i*this->batchSize)(0) , batch.n_rows, this->batchSize, false);
						arma::mat stream_labels = arma::mat(&labels.unsafe_col(i*this->batchSize)(0), 1, this->batchSize, false);
						Train(stream_points, stream_labels);
					}
					if( mod > 0 ){
						arma::mat stream_points = arma::mat(&batch.unsafe_col(batch.n_cols - mod)(0) , batch.n_rows, mod, false);
						arma::mat stream_labels = arma::mat(&labels.unsafe_col(batch.n_cols - mod)(0), 1, mod, false);
						Train(stream_points, stream_labels);
					}
				}else{
					Train(batch,labels);
				}
			}
			
		}
		
		// Get the next 1000 data points from disk to stream them.
		DataSource->advance();
		cout << "count : " << count << endl;
	}
	
	for(auto net:this->_net_container){
		net->process_fini();
	}
	count = 0;
	DataSource->rewind();
}

template<typename distrNetType>
void HDF5_Feeder< distrNetType >::Train(arma::mat& batch, arma::mat& labels){
	for(auto net:this->_net_container){
		size_t random_node = std::rand()%(net->sites.size());
		if(net->cfg().learning_algorithm == "MLP"){
			
			/// Transform the labels to the mlpack's Neural Net format.
			/// After training the transformation is reversed. 
			labels.transform( [](double val) { return (val == -1.) ? 1.: val+1.; } );
			net->process_record(random_node, batch, labels);
			labels.transform( [](double val) { return (val == 1.) ? -1.: val-1; } );
			
		}else{
			/// Train on data point/batch.
			net->process_record(random_node, batch, labels);
		}
	}
}


/**
	An HDF5 feader that provides a non-stationary stream (with concept drift).
	Only the Extreme Learning Machine Classifier is compatible with this class at the moment.
	**/
template<typename distrNetType>
class HDF5_Drift_Feeder : public feeder<distrNetType>{
	typedef boost::shared_ptr<hdf5Source<double>> PointToSource;
protected:
	PointToSource TrainSource; // Data Source to read the dataset in streaming form.
	PointToSource TestSource; // Data Source to read the dataset in streaming form.
	int experiment; // The concept drift experiment.
	int num_of_classes; // The number of classes in this concept.
public:
	/** 	
		Constructor.
		**/
	HDF5_Drift_Feeder(string cfg);
	
	/**	
	    A destructor for the class.
		Probably needs some fixing later on.
		**/
	~HDF5_Drift_Feeder();
	
	void makeTestDataset() override;
	
	arma::mat one_hot_labels(const arma::mat& lbs);
	
	void TrainNetworks();
	
	void Train(arma::mat& batch, arma::mat& labels);
	
	void getStatistics() override { }
	
	inline size_t getNumberOfFeatures() override { return TrainSource->getDataSize()-1; }
	
};

template<typename distrNetType>
HDF5_Drift_Feeder<distrNetType>::HDF5_Drift_Feeder(string cfg)
:feeder<distrNetType>(cfg){
	try{
		Json::Value root;
		std::ifstream cfgfile(this->config_file); // Parse from JSON file.
		cfgfile >> root;
		
		// Test if the learning algorithms are compatible with this feeder.
		size_t number_of_networks = root["simulations"].get("number_of_networks",0).asInt();
		for(size_t i = 1; i <= number_of_networks; i++){
			string learning_algorithm = root["gm_network_"+root["simulations"].get("net_name_"+std::to_string(i), "NoNet").asString()]
			                            .get("learning_algorithm", "NoAlgo").asString();
			if(learning_algorithm!="ELM"){
				cout << "Only the Extreme Learning Classifier learning algorithm is compatible with the HDF5_Drift_Feeder class at the moment." << endl;
				throw;
			}
		}
		
		// Simulation details.
		num_of_classes = 0;
		this->batchSize = root["tests_hdf5_Data"].get("batch_size",1).asInt64();
		if(this->batchSize<0 || this->batchSize>1000){
			cout << "Invalid batch size." << endl;
			cout << "Setting the batch size to default : 64" << endl;
			this->batchSize = 64;
		}
		this->warmupSize = root["tests_hdf5_Data"].get("warmup_size",64).asInt64();
		
		// Concept drift experiment.
		experiment = root["tests_hdf5_Data"].get("experiment",0).asInt();
		if(experiment!=1 && experiment!=2 && experiment!=3 && experiment!=4){
			cout << "Invalid experiment number." << endl;
			cout << "Initializing default experiment : 1" << endl;
			experiment = 1;
		}
		
		string train_concept;
		string test_concept;
		if(experiment==1){
			train_concept = "C1C2_Train";
			test_concept = "C1C2_Test";
		}else{
			train_concept = "C1_Train";
			test_concept = "C1_Test";
		}
		
		// Initialize data sources.
		TrainSource = getPSource<double>("TestFile2.h5",
									     train_concept,
									     false);
		TestSource = getPSource<double>("TestFile2.h5",
										test_concept,
										false);
		
		// Create the test dataset.
		makeTestDataset();
		
	}catch(...){
		cout << endl << "Something went wrong in HDF5_Drift_Feeder object construction." << endl;
		throw;
	}
}

template<typename distrNetType>
void HDF5_Drift_Feeder<distrNetType>::makeTestDataset(){
	this->testSet = arma::zeros<arma::mat>(TestSource->getDataSize()-1, TestSource->getDatasetLength()); // Test features set.
	arma::mat tempTestResponses = arma::zeros<arma::mat>(1, TestSource->getDatasetLength());
	
	size_t count = 0;
	vector<double>& buffer = TestSource->getbuffer(); // Initialize the dataset buffer.
	while(TestSource->isValid()){
		
		arma::mat batch; // Arma column wise matrix containing the data points.
		
		// Load the batch from the buffer while removing the ids.
		batch = arma::mat(&buffer[0],
						  TestSource->getDataSize(),
						  (int)TestSource->getBufferSize(),
						  false,
						  false);
		
		// Insert the batch to the test set.
		this->testSet.cols(count, count+TestSource->getBufferSize()-1) = batch.rows(0, batch.n_rows-2);
		tempTestResponses.cols(count, count+TestSource->getBufferSize()-1) = batch.row(batch.n_rows-1);
		count += TestSource->getBufferSize(); // Update the number of processed elements.
		
		// Get the next 1000 data points from disk to stream them.
		TestSource->advance();
	}
	
	tempTestResponses -= 1.;
	int num_of_cl = (int)tempTestResponses.max();
	this->testResponses = arma::zeros<arma::mat>(num_of_cl+1, tempTestResponses.n_cols);
	for(size_t i=0; i<tempTestResponses.n_cols; i++){
		this->testResponses((int)tempTestResponses(0,i), i) = 1.;
	}
}

template<typename distrNetType>
void HDF5_Drift_Feeder<distrNetType>::TrainNetworks(){
	
	vector<double>& buffer = TrainSource->getbuffer(); // Initialize the dataset buffer.
	int count = 0; // Count the number of processed elements.
	
	bool warm = false; // Variable indicating if the networks are warmed up.
	size_t degrees = 0; // Number of warmup datapoints read so far by the networks.
	
	// The communication statistics of the experiment.
	size_t total_messages = 0;
	size_t total_info_messages = 0;
	size_t total_bytes = 0;
	
	
	// Start training.
	for(size_t con=0; con<2; con++){
		while(TrainSource->isValid()){
			
			arma::mat batch; // Arma column wise matrix containing the data points.
			
			// Load the batch from the buffer while removing the ids.
			batch = arma::mat(&buffer[0],
							  TrainSource->getDataSize(),
							  TrainSource->getBufferSize(),
							  false,
							  false);
			arma::mat labels = arma::conv_to<arma::mat>::from(batch.row(batch.n_rows-1))-1.; // Get the labels.
			batch.shed_row(batch.n_rows-1); // Remove the labels from the batch.
							  
			// Update the number of processed elements.
			count += TrainSource->getBufferSize();

			size_t pointer = 0;
			size_t current_batch_sz = 0;
			while(true){
				
				// Calculating the appropriate batch size.
				if(pointer+this->batchSize > batch.n_cols){
					current_batch_sz = batch.n_cols%this->batchSize;
				}else{
					current_batch_sz = this->batchSize;
				}
				
				// Fetch the batch.
				arma::mat stream_batch = arma::mat(&batch.unsafe_col(pointer)(0), batch.n_rows, current_batch_sz, false);
				arma::mat stream_labels_ = arma::mat(&labels.unsafe_col(pointer)(0), labels.n_rows, current_batch_sz, false);
				arma::mat stream_labels = one_hot_labels(stream_labels_);
				
				// Warmup procedure.
				if(!warm){
					if(degrees+current_batch_sz <= this->warmupSize){ // A part of the warmup dataset is on the buffer.
						degrees += current_batch_sz;
						
						// Warming up the hubs.
						for(size_t i = 0; i < this->_net_container.size(); i++){
							this->_net_container.at(i)->warmup(stream_batch, stream_labels);
						}
						
						// Check if the warmup is done.
						if(degrees==this->warmupSize){
							warm = true;
							for(size_t i = 0; i < this->_net_container.size(); i++){
								this->_net_container.at(i)->end_warmup(); // Each hub initializes the first round.
							}
						}
					}else{
						warm = true;
						
						arma::mat stream_batch_1 = stream_batch.cols(0,this->warmupSize-degrees-1);
						arma::mat stream_labels_1 = stream_labels.cols(0,this->warmupSize-degrees-1);
						arma::mat stream_batch_2 = stream_batch.cols(this->warmupSize-degrees, stream_batch.n_cols-1);
						arma::mat stream_labels_2 = stream_labels.cols(this->warmupSize-degrees, stream_labels.n_cols-1);
						
						// Warming up the hubs.
						for(size_t i = 0; i < this->_net_container.size(); i++){
							this->_net_container.at(i)->warmup(stream_batch_1, stream_labels_1);
							this->_net_container.at(i)->end_warmup(); // Each hub initializes the first round.
						}
						
						// Train the nets with the remaining points.
						Train(stream_batch_2, stream_labels_2);
					}
					this->gatherDifferentialInfo(); // Gathering the info of the communication triggered by the streaming batch.
					pointer += current_batch_sz;    // Pointer increment.
					continue;
				}
				
				// Train the nets on the streaming batch.
				Train(stream_batch, stream_labels);
				
				// Gathering the info of the communication triggered by the streaming batch.
				this->gatherDifferentialInfo();
				
				// Pointer increment and termination condition.
				pointer += current_batch_sz;
				if(pointer==batch.n_cols)
					break;
			}
								   
			// Print some info. 
			// You can alter the polling rate here to tradeoff between info and speed of execution.
			if(count%100000==0)
				cout << endl << "count : " << count << endl << endl;
			if(count==1000 || count==1001000 || count%200000==0){
				for(auto net:this->_net_container){
					net->hub->Progress();
				}
			}
			
			// Get the next 1000 data points from disk to stream them.
			TrainSource->advance();
		}
		if(con !=1){
			cout << endl << "#############################################" << endl;
			cout << "CONCEPT DRIFT HAPPENING KNOW!" << endl;
			cout << "#############################################" << endl << endl;
			
			string train_concept;
			string test_concept;
			
			if(experiment==1 || experiment==4){
				train_concept = "C5_Train";
				test_concept = "C5_Test";
			}else if(experiment==2){
				train_concept = "C1C2_Train";
				test_concept = "C1C2_Test";
			}else if(experiment==3){
				train_concept = "C2_Train";
				test_concept = "C2_Test";
			}

			// Initialize data sources.
			TrainSource = getPSource<double>("TestFile2.h5",
											train_concept,
											false);
			TestSource = getPSource<double>("TestFile2.h5",
											test_concept,
											false);
			makeTestDataset();
		}
	}
	
	for(size_t net=0; net<this->_net_container.size(); net++){
		this->_net_container.at(net)->process_fini();
		
		std::ofstream myfile1; // File containing the total info of the experiment.
		std::ofstream myfile2; // File containing the differential communication info of the experiment.
		string fl_nm1 = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/Graphs/Distr_Learn_Comm_ELM_"
		                +this->_net_container.at(net)->rpc().name()+".csv";
		string fl_nm2 = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/Graphs/Distr_Learn_Diff_Comm_ELM_"
		                +this->_net_container.at(net)->rpc().name()+".csv";
		myfile1.open(fl_nm1, std::ios::app);
		myfile2.open(fl_nm2, std::ios::app);
		
		vector<size_t> com_stats = this->_net_container.at(net)->hub->Statistics();
		for(auto chnl:this->stats.at(net)){
			total_messages+=chnl->messages_received();
			if(chnl->bytes_received()>0){
				total_info_messages+=chnl->messages_received();
				total_bytes+=chnl->bytes_received();
			}
		}

		//myfile1 << "NetName,DistAlgo,LearnAlgo,Params,LocalSites, \
		              Rounds,Subrounds,Rebalances,Safezones,TotalMessages, \
					  TotalInfoMessages,TotalBytes,accuracy,threshold\n"
		myfile1 << this->_net_container.at(net)->name()
			   << "," << this->_net_container.at(net)->hub->cfg().distributed_learning_algorithm
			   << "," << this->_net_container.at(net)->hub->cfg().learning_algorithm
			   << "," << this->_net_container.at(net)->hub->global_learner->modelDimensions().at(0).n_rows
		       << "," << this->_net_container.at(net)->sites.size()
		       << "," << com_stats.at(0) 
		       << "," << com_stats.at(1) 
			   << "," << com_stats.at(2)
			   << "," << com_stats.at(3)
			   << "," << total_messages 
			   << "," << total_info_messages
			   << "," << total_bytes
			   << "," << this->_net_container.at(net)->hub->query->accuracy
			   << "," << std::to_string(this->_net_container.at(net)->hub->safe_zone->hyper().at(0))
			   << "," << std::to_string(this->_net_container.at(net)->hub->safe_zone->hyper().at(1))
			   << "," << this->_net_container.at(net)->hub->cfg().reb_mult
			   << "\n";
		myfile1.close();
		
		for(size_t inf_ind=0; inf_ind<this->differential_communication.at(net).at(0).size(); inf_ind++){
			myfile2 << this->differential_communication.at(net).at(0).at(inf_ind);
			if(inf_ind < this->differential_communication.at(net).at(0).size()-1)
				myfile2 << ",";
		}
		myfile2 << "\n";
		for(size_t inf_ind=0; inf_ind<this->differential_communication.at(net).at(1).size(); inf_ind++){
			myfile2 << this->differential_communication.at(net).at(1).at(inf_ind);
			if(inf_ind < this->differential_communication.at(net).at(1).size()-1)
				myfile2 << ",";
		}
		myfile2 << "\n";
//		for(size_t inf_ind=0; inf_ind<differential_accuracy.at(net).size(); inf_ind++){
//			myfile2 << differential_accuracy.at(net).at(inf_ind);
//			if(inf_ind<differential_accuracy.at(net).size()-1)
//				myfile2 << ",";
//		}
//		myfile2 << "\n";
		myfile2.close();
		
	}
	
}

template<typename distrNetType>
arma::mat HDF5_Drift_Feeder<distrNetType>::one_hot_labels(const arma::mat& lbs){
	int num_of_cl = (int)lbs.max();
	if(num_of_cl+1>num_of_classes)
		num_of_classes = num_of_cl+1;
	arma::mat stream_Y = arma::zeros<arma::mat>(num_of_classes, lbs.n_cols);
	for(size_t i=0; i<lbs.n_cols; i++){
		stream_Y((int)lbs(0,i), i) = 1.;
	}
	return stream_Y;
}

template<typename distrNetType>
void HDF5_Drift_Feeder<distrNetType>::Train(arma::mat& batch, arma::mat& labels){
	auto net_dist_it = this->net_dists.begin();
	for(auto net : this->_net_container){
		size_t random_node;
		if(this->uniform_distr){
			random_node = std::rand()%(net->sites.size());
		}else{
			double n = ((double) std::rand() / (RAND_MAX));
			if(n<this->B_prob){
				auto it = net_dist_it->at(0).begin();
				std::advance( it, (int)(std::rand()%(net_dist_it->at(0).size())) );
				random_node = *it;
			}else{
				auto it = net_dist_it->at(1).begin();
				std::advance( it, (int)(std::rand()%(net_dist_it->at(1).size())) );
				random_node = *it;
			}
			net_dist_it++;
		}
		net->process_record(random_node, batch, labels); /// Train on data point/batch.
	}
}

template<typename distrNetType>
HDF5_Drift_Feeder<distrNetType>::~HDF5_Drift_Feeder(){
	TrainSource = nullptr;
	TestSource = nullptr;
}


} /* End of namespace MLPACK_feeders */

namespace DLIB_feeders{
	
using namespace ml_gm_proto::DL_gm_networks;
using namespace ml_gm_proto::DL_fgm_networks;
	
/*	
A vector container for the networks.
*/
template<typename feats, typename lb, template<typename,typename> typename distrNetType>
class net_container : public vector<distrNetType<feats,lb>*>
{
public:
	using vector<distrNetType<feats,lb>*>::vector;

	inline void join(distrNetType<feats,lb>* net) { this->push_back(net); }
	inline void leave(int const i) { this->erase(this->begin()+i); }

};

/*	
	A vector container for the queries.
	*/
template<typename feats, typename lb>
class query_container : public vector<dl_continuous_query<feats,lb>*>
{
	public:
	using vector<dl_continuous_query<feats,lb>*>::vector;

	inline void join(dl_continuous_query<feats,lb>* qry) { this->push_back(qry); }
	inline void leave(int const i) { this->erase(this->begin()+i); }
	
};
	

/* 
	A feeders purpose is to synchronize the testing of the networks
	by providing the appropriate data stream to the nodes of each net.
	*/
template<typename feats, typename lb, template<typename,typename> typename distrNetType>
class DLIB_feeder{
	typedef std::vector<matrix<feats>> input_features;
	typedef std::vector<lb> input_labels;
protected:
	string config_file; // JSON file to read the hyperparameters.
	time_t seed; // The seed for the random generator.
	size_t batchSize; // The batch size.
	size_t warmupSize; // The size of the warmup dataset.
	size_t test_size; // Size of test dataset.
	input_features testSet; // Test dataset for validation of the classification algorithm.
	input_labels testResponses; // Arma row vector containing the labels of the evaluation set.
	
	net_container<feats,lb,distrNetType> _net_container; // A container for networks.
	query_container<feats,lb> _query_container; // A container for queries.
	
	// Stream Distribution
	bool uniform_distr;
	vector<vector<set<size_t>>> net_dists;
	float B_prob;
	float site_ratio;
	
	// Statistics collection.
	vector<chan_frame> stats; 
	vector<vector<vector<size_t>>> differential_communication;
	size_t msgs;
	size_t bts;
	vector<vector<double>> differential_accuracy;
	
public:
	/** 	
		Constructor, destructor.
		**/
	DLIB_feeder(string cfg);
	virtual ~DLIB_feeder();
	
	/** 	
	    Method that creates the test dataset.
		This method passes one time through the entire dataset,
		if the dataset is stored in a hdf5 file.
		**/
	virtual void makeTestDataset() { }
	
	/* Method that puts a network in the network container. */
	void addQuery(dl_continuous_query<feats,lb>* qry) { _query_container.join(qry); }
	
	/* Method that puts a network in the network container. */
	void addNet(distrNetType<feats,lb>* net) { _net_container.join(net); }
	
	/* Method initializing all the networks. */
	void initializeSimulation();
	
	/* Method that prints the star learning network for debbuging purposes. */
	void printStarNets() const;
	
	/* Method that gathers communication info after each streaming batch. */
	virtual void gatherDifferentialInfo()=0;
	
	// Getters.
	inline input_features& getTestSet() { return testSet; }
	inline input_labels& getTestSetLabels() { return testResponses; }
	inline size_t getRandomInt(size_t maxValue) { return std::rand() % maxValue; }
	
	virtual inline size_t getNumberOfFeatures() { return 0; }
	virtual void getStatistics() { }
};
	
template<typename feats, typename lb, template<typename,typename> typename distrNetType>
DLIB_feeder<feats,lb,distrNetType>::DLIB_feeder(string cfg):
config_file(cfg){
	Json::Value root;
	std::ifstream cfgfile(config_file); // Parse from JSON file.
	cfgfile >> root;
	
	long int random_seed = (long int)root["simulations"].get("seed",0).asInt64();
	if(random_seed>=0){
		seed = random_seed;
	}else{
		seed = time(&random_seed);
	}
	std::srand (seed);
	
	string learning_problem = root["simulations"].get("learning_problem","NoProblem").asString();
	if(learning_problem!= "classification" && learning_problem!= "regression"){
		cout << endl << "Incorrect learning problem given." << endl;
		cout << "Acceptable learning problems are : 'classification', 'regression'" << endl;
		throw;
	}
	
	// Get the stream distribution.
	uniform_distr = root[root["simulations"].get("stream_distribution", "No_Distribution").asString()]
					.get("uniform", true).asBool();
	if(!uniform_distr){
		B_prob = root[root["simulations"].get("stream_distribution", "No_Distribution").asString()]
				 .get("B_prob",-1.).asFloat();
		if(B_prob>1. || B_prob<0.){
			cout << "Invalid parameter B_prob. Probabilities must be in the interval [0,1]" << endl;
			throw;
		}
		site_ratio = root[root["simulations"].get("stream_distribution", "No_Distribution").asString()]
			         .get("site_ratio",-1.).asFloat();
		 if(site_ratio>1. || site_ratio<0.){
			cout << "Invalid parameter site_ratio. Ratios must be in the interval [0,1]" << endl;
			throw;
		}
	}
	
}

template<typename feats, typename lb, template<typename,typename> typename distrNetType>
DLIB_feeder<feats,lb,distrNetType>::~DLIB_feeder() { }
	
template<typename feats, typename lb, template<typename,typename> typename distrNetType>
void DLIB_feeder<feats,lb,distrNetType>::initializeSimulation(){
	
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
		
		if(learning_algorithm == "LeNet"){
			auto query = new dl_Classification_query<feats,lb>(&testSet, &testResponses, config_file, net_name);
			addQuery(query);
			cout << "Query added." << endl;
		}else{
			auto query = new dl_Regression_query<feats,lb>(&testSet, &testResponses, config_file, net_name);
			addQuery(query);
			cout << "Query added." << endl;
		}
		
		source_id number_of_nodes = (source_id)root["gm_network_"+net_name].get("number_of_local_nodes", 1).asInt64();
		set<source_id> node_ids;
		for(source_id j = 1; j <= number_of_nodes; j++){
			node_ids.insert(count+j);
		}
		count += number_of_nodes+1; // We add one because of the coordinator.
		
		auto net = new distrNetType<feats,lb>(node_ids, net_name, _query_container.at(i-1));
		addNet(net);
		cout << "Net " << net_name << " initialized." << endl << endl;
	}
	
	for(auto net:_net_container){
		// Initializing the differential communication statistics.
		stats.push_back(chan_frame(net));
		vector<vector<size_t>> dif_com;
		vector<size_t> dif_msgs;
		vector<size_t> dif_bts;
		dif_msgs.push_back(0);
		dif_bts.push_back(0);
		dif_com.push_back(dif_msgs);
		dif_com.push_back(dif_bts);
		differential_communication.push_back(dif_com);
		
		vector<double> dif_acc;
		dif_acc.push_back(0.);
		differential_accuracy.push_back(dif_acc);
		
		// Initializing the stream distributions of the sites of the network.
		if(!uniform_distr){
			set<size_t> B;
			set<size_t> B_compl;
			vector<set<size_t>> net_distr;
			for(size_t i=0; i<net->sites.size(); i++){
				B_compl.insert(i);
			}
			for(size_t i=0; i<std::floor(net->sites.size()*site_ratio); i++){
				size_t n = std::rand()%(net->sites.size());
				while(B.find(n) != B.end()){
					n = std::rand()%(net->sites.size());
				}
				B.insert(n);
				B_compl.erase(n);
			}
			net_distr.push_back(B);
			net_distr.push_back(B_compl);
			net_dists.push_back(net_distr);
		}
	}
	msgs = 0;
	bts = 0;
	cout << endl << "Netwoks initialized." << endl;
}

template<typename feats, typename lb, template<typename,typename> typename distrNetType>
void DLIB_feeder<feats,lb,distrNetType>::printStarNets() const {
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

template<typename feats, typename lb, template<typename,typename> typename distrNetType>
class LeNet_Feeder : public DLIB_feeder<feats,lb,distrNetType>{
	typedef feats input_features;
	typedef lb input_labels;
	typedef boost::shared_ptr<hdf5Source<double>> PointToSource;
protected:

	Json::Value root; // JSON file to read the hyperparameters.  [optional]
	
	size_t epochs; // Number of epochs.
	double score; // Save the score of the classifier. (To be used later for hyperparameter searching)
	
	std::vector<matrix<input_features>> training_images;
	std::vector<input_labels> training_labels;
	std::vector<matrix<input_features>> test_images;
	std::vector<input_labels> test_labels;
	
	PointToSource TrainSource; // Data Source to read the training dataset in streaming form.
	PointToSource TestSource; // Data Source to load the test dataset.
	
public:

	/* Constructor */
	LeNet_Feeder(string cfg);
	
	/* Destructor */
	~LeNet_Feeder();
	
	// A method for reading the test dataset from a hdf5 file.
	void makeTestDataset() override;
	
	void TrainNetworks();
	
	void Train(std::vector<matrix<input_features>>& batch, std::vector<input_labels>& labels);
	
	void getStatistics() override { }
	
	void gatherDifferentialInfo() override;
	
	inline size_t getNumberOfFeatures() override { return TestSource->getDataSize()-1; }
	
};

template<typename feats, typename lb, template<typename,typename> typename distrNetType>
LeNet_Feeder<feats,lb,distrNetType>::LeNet_Feeder(string cfg)
:DLIB_feeder<feats,lb,distrNetType>(cfg){
	
	try{
		Json::Value root;
		std::ifstream cfgfile(this->config_file); // Parse from JSON file.
		cfgfile >> root;
		
		this->batchSize = root["tests_hdf5_Data"].get("batch_size",64).asInt64();
		this->warmupSize = root["tests_hdf5_Data"].get("warmup_size",128).asInt64();
		
		epochs = root["tests_hdf5_Data"].get("epochs",-1).asInt64();
		if(epochs<0){
			cout << endl << "Incorrect parameter epochs." << endl;
			cout << "Acceptable epochs parameter : [positive int]" << endl;
			cout << "Epochs must be in range [1:1e+6]." << endl;
			throw;
		}
		
		// Initialize data source.
		TrainSource = getPSource<double>(root["data"].get("file_name","No file name given").asString(),
									     root["data"].get("train_dset_name","No file dataset name given").asString(),
									     false);
									   
		// Initialize data source.
		TestSource = getPSource<double>(root["data"].get("file_name","No file name given").asString(),
								        root["data"].get("test_dset_name","No file dataset name given").asString(),
								        false);
			
		this->test_size = root["tests_hdf5_Data"].get("test_size",-1).asDouble();
		if(this->test_size<1 || this->test_size>100000){
			cout << endl << "Incorrect parameter test_size." << endl;
			cout << "Acceptable test_size parameters : [int] in range:[1, 100000]" << endl;
			throw;
		}
		
		// Create the test dataset.
		makeTestDataset();
		
	}catch(...){
		cout << endl << "Something went wrong in Neural Classification feeder construction." << endl;
		throw;
	}
}

template<typename feats, typename lb, template<typename,typename> typename distrNetType>
void LeNet_Feeder<feats,lb,distrNetType>::gatherDifferentialInfo(){
	// Gathering the info of the communication triggered by the streaming batch.
	for(size_t i=0; i<this->_net_container.size(); i++){
		size_t batch_messages = 0;
		size_t batch_bytes = 0;
		
		for(auto chnl:this->stats.at(i)){
			batch_messages += chnl->messages_received();
			batch_bytes += chnl->bytes_received();
		}
		
		size_t dif1 = batch_messages - this->msgs;
		this->differential_communication.at(i).at(0)
								        .push_back( batch_messages - this->msgs );
		
		size_t dif2 = batch_bytes - this->bts;
		this->differential_communication.at(i).at(1)
								        .push_back( batch_bytes - this->bts  );
								  
		this->msgs = batch_messages;
		this->bts = batch_bytes;
		
		//differential_accuracy.at(i).push_back(_net_container.at(i)->hub->getAccuracy());
	}
}

template<typename feats, typename lb, template<typename,typename> typename distrNetType>
LeNet_Feeder<feats,lb,distrNetType>::~LeNet_Feeder(){ 
	TrainSource = nullptr;
	TestSource = nullptr;
}

template<typename feats, typename lb, template<typename,typename> typename distrNetType>
void LeNet_Feeder<feats,lb,distrNetType>::makeTestDataset(){
	vector<double>& buffer = TestSource->getbuffer(); // Initialize the dataset buffer.
	while(TestSource->isValid()){
		for(size_t i=0;i<buffer.size();){
			matrix<input_features,28,28> image;
			input_labels label;
			for(size_t j=0;j<28;j++){
				for(size_t k=0;k<28;k++){
					image(j,k)=(input_features)buffer.at(i);
					++i;
				}
			}
			label=(input_labels)(buffer.at(i)-1);
			++i;
			this->testSet.push_back(image);
			this->testResponses.push_back(label);
		}
		TestSource->advance();
	}
}

template<typename feats, typename lb, template<typename,typename> typename distrNetType>
void LeNet_Feeder<feats,lb,distrNetType>::TrainNetworks(){
	
	vector<double>& buffer = TrainSource->getbuffer(); 
	std::vector<matrix<input_features>> mini_batch_samples;
    std::vector<input_labels> mini_batch_labels;
	size_t count = 0;
	
	bool warm = false; // Variable indicating if the networks are warmed up.
	size_t degrees = 0; // Number of warmup datapoints read so far by the networks.
	
	// The communication statistics of the experiment.
	size_t total_messages = 0;
	size_t total_info_messages = 0;
	size_t total_bytes = 0;
	
	cout << endl << "Size of the Streaming Dataset : " << TrainSource->getDatasetLength() << endl << endl;
	
	for(size_t ep=0; ep<epochs; ep++){
		cout << "Starting epoch " << ep+1 << endl;
		while(TrainSource->isValid()){
			// Load the training images and labels from the buffer.
			training_images.clear();
			training_labels.clear();
			for(size_t i=0;i<buffer.size();){
				matrix<input_features,28,28> image;
				input_labels label;
				for(size_t j=0;j<28;j++){
					for(size_t k=0;k<28;k++){
						image(j,k)=(input_features)buffer.at(i);
						++i;
					}
				}
				label=(input_labels)(buffer.at(i)-1);
				++i;
				training_images.push_back(image);
				training_labels.push_back(label);
			}
								   
			// Update the number of processed elements.
			count += TrainSource->getBufferSize();
			
			// Warming up the nets if necessary.
			if(!warm){
				if(training_images.size() <= this->warmupSize-degrees){ // A part of the warmup dataset is on the buffer.
					degrees += training_images.size();
					size_t posit1=0;
					bool done1=false;
					while(posit1<training_images.size()){
						mini_batch_samples.clear();
						mini_batch_labels.clear();
						while(mini_batch_samples.size()<this->batchSize){
							if(posit1<training_images.size()){
								mini_batch_samples.push_back(training_images[posit1]);
								mini_batch_labels.push_back(training_labels[posit1]);
								posit1++;
							}else{
								break;
							}
						}
						// Warming up the hubs of the star networks.
						for(auto net:this->_net_container){
							net->warmup(mini_batch_samples, mini_batch_labels);
						}
					}
					gatherDifferentialInfo();
					TrainSource->advance();
					continue;
				}else{ // The last chunk of the warmup dataset.
					size_t posit1=0;
					bool done1=false;
					while(posit1<(this->warmupSize-degrees)){
						mini_batch_samples.clear();
						mini_batch_labels.clear();
						while(mini_batch_samples.size()<this->batchSize){
							if(posit1<(this->warmupSize-degrees)){
								mini_batch_samples.push_back(training_images[posit1]);
								mini_batch_labels.push_back(training_labels[posit1]);
								posit1++;
							}else{
								break;
							}
						}
						// Warming up the hubs of the star networks.
						for(auto net:this->_net_container){
							net->warmup(mini_batch_samples, mini_batch_labels);
						}
					}
					// Removing the warm up data from the buffer.
					for(size_t i=0; i<(this->warmupSize-degrees); i++){
						training_images.erase(training_images.begin());
						training_labels.erase(training_labels.begin());
					}
					// Initializing all the nodes. Starting the first round.
					for(auto net:this->_net_container){
						// Each hub initializes the first round.
						net->end_warmup();
					}
					gatherDifferentialInfo();
					warm = true;
				}
			}
			
			// Train the nets.
			size_t posit=0;
			bool done=false;
			while(posit<training_images.size()&&(!done)){
				// Create the batch.
				mini_batch_samples.clear();
				mini_batch_labels.clear();
				while(mini_batch_samples.size()<this->batchSize){
					if(posit<training_images.size()){
						mini_batch_samples.push_back(training_images[posit]);
						mini_batch_labels.push_back(training_labels[posit]);
						posit++;
					}else{
						done=true;
						break;
					}
				}
				
				// Train the nets on the streaming batch.
				Train(mini_batch_samples, mini_batch_labels);
				
				// Gathering the info of the communication triggered by the streaming batch.
				gatherDifferentialInfo();
				
			}

			// Get the next 1000 data points from disk to stream them.
			TrainSource->advance();
			
			// Print some info. 
			// You can alter the polling rate here to tradeoff between info and speed of execution.
			if(count%100000==0)
				cout << "count : " << count << endl << endl;
			if(count%200000==0){
				for(auto net:this->_net_container){
					net->hub->Progress();
				}
			}
			
		}
		count = 0;
		TrainSource->rewind();
	}
	
	for(size_t net=0;net<this->_net_container.size();net++){
		this->_net_container.at(net)->process_fini();
		
		std::ofstream myfile1; // File containing the total info of the experiment.
		std::ofstream myfile2; // File containing the differential communication info of the experiment.
		string fl_nm1 = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/Graphs/Distr_Learn_Comm_"
		                +this->_net_container.at(net)->rpc().name()+".csv";
		string fl_nm2 = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/Graphs/Distr_Learn_Diff_Comm_"
		                +this->_net_container.at(net)->rpc().name()+".csv";
		myfile1.open(fl_nm1, std::ios::app);
		myfile2.open(fl_nm2, std::ios::app);
		
		vector<size_t> com_stats = this->_net_container.at(net)->hub->Statistics();
		for(auto chnl:this->stats.at(net)){
			total_messages+=chnl->messages_received();
			if(chnl->bytes_received()>0){
				total_info_messages+=chnl->messages_received();
				total_bytes+=chnl->bytes_received();
			}
		}
		
		//myfile1 << "NetName,DistAlgo,LearnAlgo,Params,LocalSites,Rounds,Subrounds,Rebalances,Safezones,TotalMessages,TotalInfoMessages,TotalBytes,accuracy,threshold\n"
		myfile1 << this->_net_container.at(net)->name()
			    << "," << this->_net_container.at(net)->hub->cfg().distributed_learning_algorithm
			    << "," << this->_net_container.at(net)->hub->cfg().learning_algorithm
			    << "," << this->_net_container.at(net)->hub->global_learner->modelDimensions()
		        << "," << this->_net_container.at(net)->sites.size()
		        << "," << com_stats.at(0) 
		        << "," << com_stats.at(1) 
			    << "," << com_stats.at(2)
			    << "," << com_stats.at(3)
			    << "," << total_messages 
			    << "," << total_info_messages
			    << "," << total_bytes
			    << "," << (float)100.0*this->_net_container.at(net)->hub->query->accuracy
			    << "," << std::to_string(this->_net_container.at(net)->hub->safe_zone->hyper().at(0))
			    << "," << std::to_string(this->_net_container.at(net)->hub->safe_zone->hyper().at(1))
			    << "," << this->_net_container.at(net)->hub->cfg().reb_mult
			    << "\n";
		myfile1.close();
		
		for(size_t inf_ind=0; inf_ind<this->differential_communication.at(net).at(0).size(); inf_ind++){
			myfile2 << this->differential_communication.at(net).at(0).at(inf_ind);
			if(inf_ind < this->differential_communication.at(net).at(0).size()-1)
				myfile2 << ",";
		}
		myfile2 << "\n";
		for(size_t inf_ind=0; inf_ind<this->differential_communication.at(net).at(1).size(); inf_ind++){
			myfile2 << this->differential_communication.at(net).at(1).at(inf_ind);
			if(inf_ind < this->differential_communication.at(net).at(1).size()-1)
				myfile2 << ",";
		}
		myfile2 << "\n";
//		for(size_t inf_ind=0; inf_ind<this->differential_accuracy.at(net).size(); inf_ind++){
//			myfile2 << this->differential_accuracy.at(net).at(inf_ind);
//			if(inf_ind < this->differential_accuracy.at(net).size()-1)
//				myfile2 << ",";
//		}
//		myfile2 << "\n";
		myfile2.close();
		
	}
	count = 0;
	TrainSource->rewind();
	
}

template<typename feats, typename lbs, template<typename,typename> typename distrNetType>
void LeNet_Feeder<feats,lbs,distrNetType>::Train(std::vector<matrix<feats>>& batch, std::vector<lbs>& labels){
//	for(auto net:this->_net_container){
//		// Picking a random node to train.
//		size_t random_node = std::rand()%(net->sites.size());
//		// Train on data point/batch.
//		net->process_record(random_node, batch, labels);
//	}
	auto net_dist_it = this->net_dists.begin();
	for(auto net : this->_net_container){
		size_t random_node;
		if(this->uniform_distr){
			random_node = std::rand()%(net->sites.size());
		}else{
			double n = ((double) std::rand() / (RAND_MAX));
			if(n<this->B_prob){
				auto it = net_dist_it->at(0).begin();
				std::advance( it, (int)(std::rand()%(net_dist_it->at(0).size())) );
				random_node = *it;
			}else{
				auto it = net_dist_it->at(1).begin();
				std::advance( it, (int)(std::rand()%(net_dist_it->at(1).size())) );
				random_node = *it;
			}
			net_dist_it++;
		}
		net->process_record(random_node, batch, labels); /// Train on data point/batch.
	}
}
	
} /* End of namespace DLIB_feeders */

} /* End of namespace feeders */

#endif
