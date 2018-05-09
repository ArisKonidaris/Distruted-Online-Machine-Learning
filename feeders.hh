#ifndef __FEEDERS_HH__
#define __FEEDERS_HH__

#include "ML_GM_Networks.hh"
#include "DL_GM_Networks.hh"

namespace feeders{

using namespace ml_gm_proto;

namespace MLPACK_feeders{

using namespace ml_gm_proto::ML_gm_networks;
	
/*	
	A vector container for the networks.
	*/
class net_container : public vector<network*>
{
public:
	using vector<network*>::vector;

	inline void join(network* net) { this->push_back(net); }
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
class feeder{
protected:
	string config_file; // JSON file to read the hyperparameters.
	time_t seed; // The seed for the random generator.
	size_t batchSize; // The batch learning size.
	size_t warmupSize; // The size of the warmup dataset.
	size_t test_size; // Size of test dataset.
	bool negative_labels; // If true it sets 0 labels to -1.
	arma::mat testSet; // Test dataset for validation of the classification algorithm.
	arma::mat testResponses; // Arma row vector containing the labels of the evaluation set.
	
	net_container _net_container; // A container for networks.
	query_container _query_container; // A container for queries.
	
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
	void addNet(network* net) { _net_container.join(net); }
	
	/* Method initializing all the networks. */
	void initializeSimulation();
	
	/* Method that prints the star learning network for debbuging purposes. */
	void printStarNets() const;
	
	// Getters.
	inline arma::mat& getTestSet() { return testSet; }
	inline arma::mat& getTestSetLabels() { return testResponses; }
	inline arma::mat* getPTestSet() { return &testSet; }
	inline arma::mat* getPTestSetLabels() { return &testResponses; }
	inline size_t getRandomInt(size_t maxValue) { return std::rand() % maxValue; }
	
	virtual inline size_t getNumberOfFeatures() { return 0; }
	virtual void getStatistics() { }
};

class Random_Feeder : public feeder{
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

template<typename DtTp>
class HDF5_Feeder : public feeder{
	typedef boost::shared_ptr<hdf5Source<DtTp>> PointToSource;
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

template<typename DtTp>
HDF5_Feeder< DtTp >::HDF5_Feeder(string cfg)
:feeder(cfg){
	try{
		Json::Value root;
		std::ifstream cfgfile(config_file); // Parse from JSON file.
		cfgfile >> root;
		
		batchSize = root["tests_hdf5_Data"].get("batch_size",1).asInt64();
		warmupSize = root["tests_hdf5_Data"].get("warmup_size",500).asInt64();
		
		// Initialize data source.
		DataSource = getPSource<DtTp>(root["data"].get("file_name","No file name given").asString(),
									  root["data"].get("dataset_name","No file dataset name given").asString(),
									  false);
			
		double test_sz = root["tests_hdf5_Data"].get("test_size",-1).asDouble();
		if(test_sz<0.0 || test_sz>1.0){
			cout << endl << "Incorrect parameter test_size." << endl;
			cout << "Acceptable test_size parameters : [double] in range:[0.0, 1.0]" << endl;
			throw;
		}
						
		// Determining the size of the test Dataset.
		test_size = std::floor(test_sz*DataSource->getDatasetLength());
		std::srand (seed);
		start_test = std::rand()%(DataSource->getDatasetLength()-test_size+1);
		
		// Initializing the test dataset and its labels with zeros.
		testSet = arma::zeros<arma::mat>(DataSource->getDataSize(),test_size);
		testResponses = arma::zeros<arma::mat>(1,test_size);
		
		cout << endl << "start_test : " << start_test << endl;
		cout << "test_size : " << test_size << endl << endl;
		
		// Create the test dataset.
		makeTestDataset();
		
	}catch(...){
		cout << endl << "Something went wrong in HDF5_Feeder object construction." << endl;
		throw;
	}
}

template<typename DtTp>
HDF5_Feeder< DtTp >::~HDF5_Feeder(){
	DataSource = nullptr;
}

template<typename DtTp>
void HDF5_Feeder< DtTp >::makeTestDataset(){
	vector<DtTp>& buffer = DataSource->getbuffer(); // Initialize the dataset buffer.
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
		
		if(count>=start_test && tests_points_read!=test_size){
			
			if(tests_points_read==0){
				start = start_test - (count-DataSource->getBufferSize()+1);
			}
			
			if(count < start_test + test_size -1){
				
				testSet.cols(tests_points_read, tests_points_read+DataSource->getBufferSize()-1-start) = 
				batch.cols(start, batch.n_cols-1); // Create the test dataset.
							 
				testResponses.cols(tests_points_read, tests_points_read+DataSource->getBufferSize()-1-start) =
				arma::conv_to<arma::mat>::from(testSet.cols(tests_points_read, tests_points_read+DataSource->getBufferSize()-1-start)
													  .row(batch.n_rows-1)); // Get the labels.
				
				batch.shed_cols(start, batch.n_cols-1); // Remove the test dataset from the batch.
				
				tests_points_read += DataSource->getBufferSize() - start;
				start = 0;
				
			}else{
				
				testSet.cols(tests_points_read, test_size-1) = 
				batch.cols(start, start+(test_size-tests_points_read)-1); // Create the test dataset.
															
				testResponses.cols(tests_points_read, test_size-1) = 
				arma::conv_to<arma::mat>::from(testSet.cols(tests_points_read, test_size-1)
													  .row(batch.n_rows-1)); // Get the labels.
				
				batch.shed_cols(start, start+(test_size-tests_points_read)-1); // Remove the test dataset from the batch.
				
				tests_points_read = test_size;
				
			}
			
			if(tests_points_read == test_size){
				testSet.shed_row(testSet.n_rows-1);
				// Replace every -1 label with 1 if necessary.
				if(negative_labels)
					testResponses.transform( [](double val) { return (val == 0.) ? -1.: val; } );
				break;
			}
			
		}
		
		// Get the next 5000 data points from disk to stream them.
		DataSource->advance();
	}
	DataSource->rewind();
}

template<typename DtTp>
void HDF5_Feeder< DtTp >::TrainNetworks(){
	vector<DtTp>& buffer = DataSource->getbuffer(); // Initialize the dataset buffer.
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
		if(count>=start_test && tests_points_read!=test_size){
			if(tests_points_read==0){
				start = start_test - (count-DataSource->getBufferSize()+1);
			}
			if(count < start_test + test_size -1){
				batch.shed_cols(start, batch.n_cols-1); // Remove the test dataset from the batch.
				tests_points_read += DataSource->getBufferSize() - start;
				start = 0;
			}else{
				batch.shed_cols(start, start+(test_size-tests_points_read)-1); // Remove the test dataset from the batch.
				tests_points_read = test_size;
			}
		}
		
		// Train the nets
		if(batch.n_cols!=0){
			
			arma::mat labels = arma::conv_to<arma::mat>::from(batch.row(batch.n_rows-1)); // Get the labels.
			batch.shed_row(batch.n_rows-1); // Remove the labels.
			if(negative_labels){ // Replace every -1 label with 1 if necessary.
				labels.transform( [](double val) { return (val == 0.) ? -1.: val; } );
			}
			
			if(!warm){ // We warm up the networks.
				if(batch.n_cols <= warmupSize-degrees){ // A part of the warmup dataset is on the buffer.
					degrees += batch.n_cols;
					for(size_t i = 0; i < _net_container.size(); i++){
						if(_net_container.at(i)->Q->config.learning_algorithm == "MLP"){
							labels.transform( [](double val) { return (val == -1.) ? 1.: val+1.; } );
							_net_container.at(i)->warmup(batch,labels);
							labels.transform( [](double val) { return (val == 1.) ? -1.: val-1; } );
						}else{
							_net_container.at(i)->warmup(batch,labels);
						}
					}
					continue;
				}else{ // The last chunk of the warmup dataset.
					arma::mat stream_points = arma::mat(&batch.unsafe_col(0)(0) , batch.n_rows, warmupSize-degrees, false);
					arma::mat stream_labels = arma::mat(&labels.unsafe_col(0)(0), 1, warmupSize-degrees, false);
					
					warm = true;
					for(size_t i = 0; i < _net_container.size(); i++){
						if(_net_container.at(i)->Q->config.learning_algorithm == "MLP"){
							stream_labels.transform( [](double val) { return (val == -1.) ? 1.: val+1.; } );
							_net_container.at(i)->warmup(stream_points, stream_labels);
							stream_labels.transform( [](double val) { return (val == 1.) ? -1.: val-1; } );
						}else{
							_net_container.at(i)->warmup(stream_points, stream_labels);
						}
						_net_container.at(i)->start_round(); // Each hub initializes the first round.
					}
					
					// Remove the warmup dataset from the buffer.
					batch.shed_cols(0,warmupSize-degrees-1);
					labels.shed_cols(0,warmupSize-degrees-1);
				}
			}
			
			if(batch.n_cols!=0){ // The networks are warmed up so we train.
				// Do batch learning. The batch can be an integer in the interval [1:5000].
				size_t mod = batch.n_cols % batchSize;
				size_t num_of_batches = std::floor( batch.n_cols / batchSize );
				
				if( num_of_batches > 0 ){
					for(unsigned i = 0; i < num_of_batches; ++i){
						arma::mat stream_points = arma::mat(&batch.unsafe_col(i*batchSize)(0) , batch.n_rows, batchSize, false);
						arma::mat stream_labels = arma::mat(&labels.unsafe_col(i*batchSize)(0), 1, batchSize, false);
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
		
		// Get the next 5000 data points from disk to stream them.
		DataSource->advance();
		cout << "count : " << count << endl;
	}
	
	for(auto net:_net_container){
		net->process_fini();
	}
	count = 0;
	DataSource->rewind();
}

template<typename DtTp>
void HDF5_Feeder< DtTp >::Train(arma::mat& batch, arma::mat& labels){
	for(auto net:_net_container){
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
	
} /* End of namespace MLPACK_feeders */

namespace DLIB_feeders{
	
using namespace ml_gm_proto::DL_gm_networks;
	
/*	
A vector container for the networks.
*/
template<typename feats,typename lb>
class net_container : public vector<network<feats,lb>*>
{
public:
	using vector<network<feats,lb>*>::vector;

	inline void join(network<feats,lb>* net) { this->push_back(net); }
	inline void leave(int const i) { this->erase(this->begin()+i); }

};

/*	
	A vector container for the queries.
	*/
template<typename feats,typename lb>
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
template<typename feats, typename lb>
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
	
	net_container<feats,lb> _net_container; // A container for networks.
	query_container<feats,lb> _query_container; // A container for queries.
	
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
	void addNet(network<feats,lb>* net) { _net_container.join(net); }
	
	/* Method initializing all the networks. */
	void initializeSimulation();
	
	/* Method that prints the star learning network for debbuging purposes. */
	void printStarNets() const;
	
	// Getters.
	inline input_features& getTestSet() { return testSet; }
	inline input_labels& getTestSetLabels() { return testResponses; }
	inline size_t getRandomInt(size_t maxValue) { return std::rand() % maxValue; }
	
	virtual inline size_t getNumberOfFeatures() { return 0; }
	virtual void getStatistics() { }
};
	
template<typename feats, typename lb>
DLIB_feeder<feats,lb>::DLIB_feeder(string cfg):
config_file(cfg){
	Json::Value root;
	std::ifstream cfgfile(config_file); // Parse from JSON file.
	cfgfile >> root;
	
	long int random_seed = (long int)root["simulations"].get("seed",0).asInt64();
	seed = time(&random_seed);
	
	string learning_problem = root["simulations"].get("learning_problem","NoProblem").asString();
	if(learning_problem!= "classification" && learning_problem!= "regression"){
		cout << endl << "Incorrect learning problem given." << endl;
		cout << "Acceptable learning problems are : 'classification', 'regression'" << endl;
		throw;
	}
}

template<typename feats, typename lb>
DLIB_feeder<feats,lb>::~DLIB_feeder() { }
	
template<typename feats, typename lb>
void DLIB_feeder<feats,lb>::initializeSimulation(){
	
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
		
		auto net = new network<feats,lb>(node_ids, net_name, _query_container.at(i-1));
		addNet(net);
		cout << "Net " << net_name << " initialized." << endl << endl;
	}
	cout << endl << "Netwoks initialized." << endl;
}

template<typename feats, typename lb>
void DLIB_feeder<feats,lb>::printStarNets() const {
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

template<typename feats, typename lb>
class LeNet_Feeder : public DLIB_feeder<feats,lb>{
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
	
	inline size_t getNumberOfFeatures() override { return TestSource->getDataSize()-1; }
	
};

template<typename feats, typename lb>
LeNet_Feeder<feats,lb>::LeNet_Feeder(string cfg)
:DLIB_feeder<feats,lb>(cfg){
	
	try{
		std::srand (this->seed);
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

template<typename feats, typename lb>
LeNet_Feeder<feats,lb>::~LeNet_Feeder(){ 
	TrainSource = nullptr;
	TestSource = nullptr;
}

template<typename feats, typename lb>
void LeNet_Feeder<feats,lb>::makeTestDataset(){
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

template<typename feats, typename lb>
void LeNet_Feeder<feats,lb>::TrainNetworks(){
	
	vector<double>& buffer = TrainSource->getbuffer(); 
	std::vector<matrix<input_features>> mini_batch_samples;
    std::vector<input_labels> mini_batch_labels;
	size_t count = 0;
	bool warm = false; // Variable indicating if the networks are warmed up.
	size_t degrees = 0; // Number of warmup datapoints read so far by the networks.
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
			if(!warm){ // We warm up the networks.
				if(training_images.size() <= this->warmupSize-degrees){ // A part of the warmup dataset is on the buffer.
					degrees += training_images.size();
					size_t posit1=0;
					bool done1=false;
					while(posit1<training_images.size()&&(!done1)){
						mini_batch_samples.clear();
						mini_batch_labels.clear();
						while(mini_batch_samples.size()<this->batchSize){
							if(posit1<training_images.size()){
								mini_batch_samples.push_back(training_images[posit1]);
								mini_batch_labels.push_back(training_labels[posit1]);
								posit1++;
							}else{
								done1=true;
								break;
							}
						}
						for(auto net:this->_net_container){
							// Picking a random node to train.
							for(size_t i=0;i<net->sites.size();i++){
								net->warmup(i, mini_batch_samples, mini_batch_labels);
							}
						}
					}
					continue;
				}else{ // The last chunk of the warmup dataset.
					size_t posit1=0;
					bool done1=false;
					while(posit1<(this->warmupSize-degrees)&&(!done1)){
						mini_batch_samples.clear();
						mini_batch_labels.clear();
						while(mini_batch_samples.size()<this->batchSize){
							if(posit1<(this->warmupSize-degrees)){
								mini_batch_samples.push_back(training_images[posit1]);
								mini_batch_labels.push_back(training_labels[posit1]);
								posit1++;
							}else{
								done1=true;
								break;
							}
						}
						for(auto net:this->_net_container){
							// Picking a random node to train.
							for(size_t i=0;i<net->sites.size();i++){
								net->warmup(i, mini_batch_samples, mini_batch_labels);
							}
						}
					}
					// Removing the warm up data from the buffer.
					for(size_t i=0; i<(this->warmupSize-degrees); i++){
						training_images.erase(training_images.begin());
						training_labels.erase(training_labels.begin());
					}
					for(auto net:this->_net_container){
						// Each hub initializes the first round.
						for(size_t i=0;i<net->sites.size();i++){
							net->end_warmup(i);
						}
					}
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
				Train(mini_batch_samples,mini_batch_labels);
			}

			// Get the next 5000 data points from disk to stream them.
			TrainSource->advance();
			cout << "count : " << count << endl;
			if(count%10000==0){
				for(auto net:this->_net_container){
					net->hub->Progress();
				}
			}
			
		}
		count = 0;
		TrainSource->rewind();
	}
	
	for(auto net:this->_net_container){
		net->process_fini();
	}
	count = 0;
	TrainSource->rewind();
}

template<typename feats, typename lbs>
void LeNet_Feeder<feats,lbs>::Train(std::vector<matrix<feats>>& batch, std::vector<lbs>& labels){
	for(auto net:this->_net_container){
		// Picking a random node to train.
		size_t random_node = std::rand()%(net->sites.size());
		// Train on data point/batch.
		net->process_record(random_node, batch, labels);
	}
}
	
} /* End of namespace DLIB_feeders */




}

#endif
