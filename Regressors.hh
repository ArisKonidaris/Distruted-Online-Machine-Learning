#ifndef _REGRESSORS_HH_
#define _REGRESSORS_HH_

#include <iostream>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <random>
#include <algorithm>
#include <jsoncpp/json/json.h>
#include <cmath>
#include <random>
#include <time.h>
#include "dsource.hh"

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/core/optimizers/ada_grad/ada_grad.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/core/optimizers/adam/adam.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

using std::cout;
using std::endl;
using std::vector;
using namespace data_src;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

namespace ML_Regression{
	
	class Regression{
	public:
		virtual void Train() = 0;
	};
	
	template<typename clsfr, typename DtTp>
	class Simple_Regression : public Regression{
	protected:

		Json::Value root; // JSON file to read the hyperparameters.  [optional]
		
		H5std_string Filename; // The name of file containing the data.
		H5std_string Dataset_Name; // The name of the dataset in the file.
		hsize_t Data_Size; // Number of features plus id and label.
		hsize_t Number_Of_Points; // Number of data points.
		
		clsfr* Regressor; // The classifier.
		int test_size; // Size of test dataset.
		int start_test; // Starting test data point;
		int epochs; // Number of epochs.
		arma::mat testSet; // Test dataset for validation of the classification algorithm.
		arma::dvec testResponses; // Arma row vector containing the labels of the evaluation set.
		double RMSE; // The root mean squared error of the regressor on the test dataset.
		
		boost::shared_ptr<hdf5Source<DtTp>> DataSource; // Data Source to read the dataset in streaming form.
		
	public:
		Simple_Regression(string cfg){
			
			try{
				
				// Parse from JSON file.
				std::ifstream cfgfile(cfg);
				cfgfile >> root;
				
				double test_sz = root["parameters"].get("test_size",-1).asDouble();
				if(test_sz<0.0 || test_sz>1.0){
					cout << endl << "Incorrect parameter test_size." << endl;
					cout << "Acceptable test_size parameters : [double] in range:[0.0, 1.0]" << endl;
					throw;
				}
				
				epochs = root["parameters"].get("epochs",-1).asInt64();
				if(epochs<0){
					cout << endl << "Incorrect parameter epochs." << endl;
					cout << "Acceptable epochs parameters : [positive int]" << endl;
					cout << "Epochs must be in range [1:1e+6]." << endl;
					throw;
				}
				
				Filename = root["data"].get("file_name","No file name given").asString();
				Dataset_Name = root["data"].get("dataset_name","No file dataset name given").asString();
				
				// Initialize the Regressor object.
				Regressor = new clsfr(cfg);

				// Initialize data source.
				DataSource = getPSource<DtTp>(Filename,
											  Dataset_Name,
											  false);
											
				test_size = std::floor(test_sz*DataSource->getDatasetLength());
				std::srand (time(NULL));
				start_test = std::rand()%(DataSource->getDatasetLength()-test_size+1);
				
				testSet = arma::zeros<arma::mat>(DataSource->getDataSize(),test_size); // Test dataset for validation of the classification algorithm.
				testResponses = arma::zeros<arma::dvec>(test_size); // Arma row vector containing the labels of the evaluation set.
				
			}catch(...){
				cout << endl << "Something went wrong in LinearClassification object construction." << endl;
				throw;
			}
		};
		
		~Simple_Regression() { delete DataSource; }
		
		// Begin the training process.
		void Train() override;
		
		// Make a prediction.
		inline arma::dvec MakePrediction(arma::mat& batch){ return Regressor->predict(batch);};
		
		// Make a prediction.
		inline double MakePrediction(arma::dvec& data_point){return Regressor->predict(data_point);};
		
		// Get score.
		void getScore(arma::mat& test_data, arma::dvec& labels);
		
		// Return the Root Mean Squared Error.
		double& getRMSE(arma::mat& test_data, arma::dvec& labels);
		
		// Print the predictions alongside the true labels.
		void printPredictions(arma::mat& test_data, arma::dvec& labels);
		
		// Getters.
		inline arma::mat& getTestSet(){return testSet;};
		inline arma::dvec& getTestSetLabels(){return testResponses;};
		
	};
	
	template<typename clsfr, typename DtTp>
	void Simple_Regression< clsfr, DtTp >::Train(){
		vector<DtTp>& buffer = DataSource->getbuffer(); // Initialize the dataset buffer.
		int count = 0; // Count the number of processed elements.
		
		for(int ep = 0; ep < epochs; ep++){
			int tests_points_read = 0;
			int start = 0;
			testSet = arma::zeros<arma::mat>(DataSource->getDataSize(),test_size); // Test dataset for validation of the classification algorithm.
			testResponses = arma::zeros<arma::dvec>(test_size); // Arma row vector containing the labels of the evaluation set.
			while(DataSource->isValid()){
			
				arma::mat batch; // Arma column wise matrix containing the data points.
				arma::dvec labels; // Arma row vector containing the labels of the training set.
				
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
						testResponses.rows(tests_points_read, tests_points_read+DataSource->getBufferSize()-1-start) = 
						arma::conv_to<arma::dvec>::from(trans(testSet.cols(tests_points_read, tests_points_read+DataSource->getBufferSize()-1-start)
																	 .row(batch.n_rows-1))); // Get the labels.
						
						batch.shed_cols(start, batch.n_cols-1); // Remove the test dataset from the batch.
						
						tests_points_read += DataSource->getBufferSize() - start;
						start = 0;
						
					}else{
						
						testSet.cols(tests_points_read, test_size-1) = 
						batch.cols(start, start+(test_size-tests_points_read)-1); // Create the test dataset.
						testResponses.rows(tests_points_read, test_size-1) = 
						arma::conv_to<arma::dvec>::from(trans(testSet.cols(tests_points_read, test_size-1)
																	 .row(batch.n_rows-1))); // Get the labels.
						
						batch.shed_cols(start, start+(test_size-tests_points_read)-1); // Remove the test dataset from the batch.
						
						tests_points_read = test_size;
						
					}
					
					if(tests_points_read == test_size){
						testSet.shed_row(testSet.n_rows-1);
					}
					
				}
				
				if(batch.n_cols!=0){
					
					labels = arma::conv_to<arma::dvec>::from(trans(batch.row(batch.n_rows-1))); // Get the labels.
					batch.shed_row(batch.n_rows-1); // Remove the labels.
					
					// Starting the online learning on the butch
					Regressor->fit(batch,labels);
					
				}
				
				// Get the next 1000 data points from disk to stream them.
				DataSource->advance();
				cout << "count : " << count << endl;
			}
			
			count = 0;
			DataSource->rewind();
		}
		
	};
	
	template<typename clsfr, typename DtTp>
	void Simple_Regression< clsfr, DtTp >::getScore(arma::mat& test_data, arma::dvec& labels){
		RMSE = Regressor->RMSE(test_data,labels);
		cout << endl << "tests : " << labels.n_elem << endl;
		cout << "start test : " << start_test << endl;
		cout << "RMSE : " << std::setprecision(6) << RMSE << endl;
	};
	
	template<typename clsfr, typename DtTp>
	double& Simple_Regression< clsfr, DtTp >::getRMSE(arma::mat& test_data, arma::dvec& labels){
		this->getScore(test_data,labels);
		return RMSE;
	}
	
	template<typename clsfr, typename DtTp>
	void Simple_Regression< clsfr, DtTp >::printPredictions(arma::mat& test_data, arma::dvec& labels){
		cout << endl << std::setprecision(6) << labels.t() << endl;
		cout << Regressor->predict(test_data).t() << endl;
	};
	
	template<typename clsfr, typename DtTp>
	class NN_Regression : public Regression{
	protected:

		Json::Value root; // JSON file to read the hyperparameters.  [optional]
		
		boost::shared_ptr<hdf5Source<DtTp>> DataSource; // Data Source to read the dataset in streaming form.
		H5std_string Filename; // The name of file containing the data.
		H5std_string Dataset_Name; // The name of the dataset in the file.
		hsize_t Data_Size; // Number of features plus id and label.
		hsize_t Number_Of_Points; // Number of data points.
		
		clsfr* Regressor; // The classifier.
		int test_size; // Size of test dataset.
		int start_test; // Starting test data point;
		int epochs; // Number of epochs.
		arma::mat testSet; // Test dataset for validation of the classification algorithm.
		arma::dvec tempTestResponses; // Arma row vector containing the labels of the evaluation set.
		arma::mat testResponses; // Arma row vector containing the labels of the evaluation set.
		double RMSE; // The root mean squared error of the regressor on the test dataset.
		
	public:
		NN_Regression(string cfg){
			
			try{
				
				// Parse from JSON file.
				std::ifstream cfgfile(cfg);
				cfgfile >> root;
				
				double test_sz = root["parameters"].get("test_size",-1).asDouble();
				if(test_sz<0.0 || test_sz>1.0){
					cout << endl << "Incorrect parameter test_size." << endl;
					cout << "Acceptable test_size parameters : [double] in range:[0.0, 1.0]" << endl;
					throw;
				}
				
				epochs = root["parameters"].get("epochs",-1).asInt64();
				if(epochs<0){
					cout << endl << "Incorrect parameter epochs." << endl;
					cout << "Acceptable epochs parameters : [positive int]" << endl;
					cout << "Epochs must be in range [1:1e+6]." << endl;
					throw;
				}
				
				Filename = root["data"].get("file_name","No file name given").asString();
				Dataset_Name = root["data"].get("dataset_name","No file dataset name given").asString();
				
				// Initialize the Regressor object.
				Regressor = new clsfr(cfg);

				// Initialize data source.
				DataSource = getPSource<DtTp>(Filename,
											  Dataset_Name,
											  false);
											
				test_size = std::floor(test_sz*DataSource->getDatasetLength());
				std::srand (time(NULL));
				start_test = std::rand()%(DataSource->getDatasetLength()-test_size+1);
				
				testSet = arma::zeros<arma::mat>(DataSource->getDataSize(),test_size); // Test dataset for validation of the classification algorithm.
				tempTestResponses = arma::zeros<arma::dvec>(test_size); // Arma row vector containing the labels of the evaluation set.
				
			}catch(...){
				cout << endl << "Something went wrong in LinearClassification object construction." << endl;
				throw;
			}
		};
		
		~NN_Regression() { delete DataSource; }
		
		// Begin the training process.
		void Train() override;
		
		// Make a prediction.
		inline arma::mat MakePredictionOnBatch(arma::mat& batch){ return Regressor->predictOnBatch(batch); };
		
		// Make a prediction.
		inline double MakePrediction(arma::mat& data_point){ return Regressor->predict(data_point); };
		
		// Get score.
		void getScore(arma::mat& test_data, arma::mat& labels);
		
		// Return the Root Mean Squared Error.
		double& getRMSE(arma::mat& test_data, arma::mat& labels);
		
		// Print the predictions alongside the true labels.
		void printPredictions(arma::mat& test_data, arma::mat& labels);
		
		// Getters.
		inline arma::mat& getTestSet() {return testSet;};
		inline arma::mat& getTestSetLabels() {return testResponses;};
		
	};
	
	template<typename clsfr, typename DtTp>
	void NN_Regression< clsfr, DtTp >::Train(){
		vector<DtTp>& buffer = DataSource->getbuffer(); // Initialize the dataset buffer.
		int count = 0; // Count the number of processed elements.
		
		for(int ep = 0; ep < epochs; ep++){
			int tests_points_read = 0;
			int start = 0;
			testSet = arma::zeros<arma::mat>(DataSource->getDataSize(),test_size); // Test dataset for validation of the classification algorithm.
			tempTestResponses = arma::zeros<arma::dvec>(test_size); // Arma row vector containing the labels of the evaluation set.
			while(DataSource->isValid()){
			
				arma::mat batch; // Arma column wise matrix containing the data points.
				arma::mat labels; // Arma row vector containing the labels of the training set.
				
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
						tempTestResponses.rows(tests_points_read, tests_points_read+DataSource->getBufferSize()-1-start) = 
						arma::conv_to<arma::dvec>::from(trans(testSet.cols(tests_points_read, tests_points_read+DataSource->getBufferSize()-1-start)
																	 .row(batch.n_rows-1))); // Get the labels.
						
						batch.shed_cols(start, batch.n_cols-1); // Remove the test dataset from the batch.
						
						tests_points_read += DataSource->getBufferSize() - start;
						start = 0;
						
					}else{
						
						testSet.cols(tests_points_read, test_size-1) = 
						batch.cols(start, start+(test_size-tests_points_read)-1); // Create the test dataset.
						tempTestResponses.rows(tests_points_read, test_size-1) = 
						arma::conv_to<arma::dvec>::from(trans(testSet.cols(tests_points_read, test_size-1)
																	 .row(batch.n_rows-1))); // Get the labels.
						
						batch.shed_cols(start, start+(test_size-tests_points_read)-1); // Remove the test dataset from the batch.
						
						tests_points_read = test_size;
						
					}
					
					if(tests_points_read == test_size){
						testSet.shed_row(testSet.n_rows-1);
						testResponses = arma::conv_to<arma::mat>::from(trans(tempTestResponses));
					}
					
				}
				
				if(batch.n_cols!=0){
					
					labels = arma::conv_to<arma::mat>::from(batch.row(batch.n_rows-1)); // Get the labels.
					batch.shed_row(batch.n_rows-1); // Remove the labels.
					
					// Starting the online learning on the batch.
					Regressor->fit(batch,labels);
					
				}
				
				// Get the next 1000 data points from disk to stream them.
				DataSource->advance();
				cout << "count : " << count << endl;
			}
			
			count = 0;
			DataSource->rewind();
		}
		
	};
	
	template<typename clsfr, typename DtTp>
	void NN_Regression< clsfr, DtTp >::getScore(arma::mat& test_data, arma::mat& labels){
		RMSE = Regressor->RMSE(test_data, labels); 
		cout << endl << "tests : " << labels.n_cols << endl;
		cout << "start test : " << start_test << endl;
		cout << "RMSE : " << std::setprecision(6) << RMSE << endl;
	};
	
	template<typename clsfr, typename DtTp>
	double& NN_Regression< clsfr, DtTp >::getRMSE(arma::mat& test_data, arma::mat& labels){
		this->getScore(test_data,labels);
		return RMSE;
	}
	
	template<typename clsfr, typename DtTp>
	void NN_Regression< clsfr, DtTp >::printPredictions(arma::mat& test_data, arma::mat& labels){
		cout << endl << std::setprecision(6) << labels << endl;
		cout << Regressor->predictOnBatch(test_data) << endl;
	};
	
	class PassiveAgressiveRegression{
	protected:
		string name = "Passive Aggressive Regressor.";
		arma::dvec W; // Parameter vector.
		string regularization; // Regularization technique.
		double C; // Regularization C term. [optional]
		Json::Value root; // JSON file to read the hyperparameters.  [optional]
		double epsilon; // The epsilon-insensitive parameter.
	public:
		PassiveAgressiveRegression(string reg, double c, double eps){
		
			try{
				
				if(eps<=0){
					cout << endl << "Invalid parameter epsilon." << endl;
					cout << "Parameter epsilon must be a positive double." << endl;
					throw;
				}
				epsilon = eps;
				
				if( reg!="none" &&
				    reg!="l1" &&
				    reg!="l2"){
					cout << endl << "Incorrect regularization parameter." << endl;
					cout << "Acceptable regularization parameters : ['none','l1','l2']" << endl;
					throw;
				}
				regularization = regularization;
				
				if(reg=="none"){
					C = 0.0;
				}else{
					if(c<0){
						cout << endl << "Incorrect parameter C." << endl;
						cout << endl << "Acceptable C parameters : [positive double]" << endl;
						throw;
					}
					C=c;
				}
				
			}catch(...){
				throw;
			}
			
		};
		PassiveAgressiveRegression(double eps):
		regularization("none"),
		C(0.0){
			try{
				if(eps<=0){
					cout << endl << "Invalid parameter epsilon." << endl;
					cout << "Parameter epsilon must be a positive double." << endl;
					throw;
				}
				epsilon = eps;
			}catch(...){
				throw;
			}
		};
		PassiveAgressiveRegression(string cfg){
			
			try{
				std::ifstream cfgfile(cfg);
				cfgfile >> root;
				regularization = root["parameters"]
								.get("regularization","Incorrect regularization parameter").asString();
				if(regularization=="Incorrect regularization parameter" ||
				   (regularization!="none" &&
				    regularization!="l1" &&
				    regularization!="l2")){
					cout << endl << "Incorrect regularization parameter." << endl;
					cout << "Acceptable regularization parameters : ['none','l1','l2']" << endl;
					throw;
				}
				
				if(regularization!="none"){
					C = root["parameters"].get("C",-1).asDouble();
					if(C<0){
						cout << endl << "Incorrect parameter C." << endl;
						cout << endl << "Acceptable C parameters : [positive double]" << endl;
						throw;
					}
				}
				
				epsilon = root["parameters"]
						  .get("epsilon",0.1).asDouble();
				if(epsilon<=0.){
					cout << endl << "Invalid parameter epsilon." << endl;
					cout << "Parameter epsilon must be a positive double." << endl;
					throw;
				}
				
			}catch(...){
				throw;
			}
			
		};
		
		// Stream update
		void fit(arma::mat& batch, arma::dvec& labels);
		
		// Make a prediction.
		arma::dvec predict(arma::mat& batch);
		
		// Make a prediction.
		double predict(arma::dvec& data_point);
		
		// Get score
		double RMSE(arma::mat& testbatch, arma::dvec& labels);
		
		// Get the type of the regressor.
		string getName(){return name;};
		
		// Get the learned parameters.
		arma::dvec& getModel(){return W;};
		
		// Get the regularization method.
		string getRegularization(){return regularization;};
		
		// Get the C regularizatio parameter.
		double getC(){ (regularization=="none") ? 0.0 : C;};
		
	};
	
	void PassiveAgressiveRegression::fit(arma::mat& batch, arma::dvec& labels){
		// Initialize the parameter vector W if has no yet been initialized.
		if (W.n_rows==0){
			W = arma::zeros<arma::dvec>(batch.n_rows);
		}
		
		// Starting the online learning
		for(int i=0; i<batch.n_cols; i++){
			
			// calculate the epsilon-insensitive loss.
			double pred = arma::dot(W,batch.unsafe_col(i) );
			double loss = std::max( 0.0 , std::abs( labels(i) - pred) - epsilon );
				
			// Calculate the Lagrange Multiplier.
			double Lagrange_Muliplier;
			if(regularization=="none"){
				Lagrange_Muliplier = loss / arma::dot(batch.unsafe_col(i),batch.unsafe_col(i)) ;
			}else if (regularization=="l1"){
				Lagrange_Muliplier = std::min(C,loss / arma::dot(batch.unsafe_col(i),batch.unsafe_col(i)));
			}else if (regularization=="l2"){
				Lagrange_Muliplier = loss / ( arma::dot(batch.unsafe_col(i),batch.unsafe_col(i)) + 1/(2*C) );
			}else{
				//throw exception
				cout << "throw exception" << endl;
			}
			
			double sign=1.0;
			if (labels(i) - pred < 0){
				sign=-1.0;
			}
			
			// Update the parameters.
			W += Lagrange_Muliplier*sign*batch.unsafe_col(i);
		}
		
	};
	
	arma::dvec PassiveAgressiveRegression::predict(arma::mat& batch){
		arma::dvec prediction = arma::zeros<arma::dvec>(batch.n_cols);
		for(int i=0;i<batch.n_cols;i++){
			prediction(i) = arma::dot(W,batch.unsafe_col(i));
		}
		
		return prediction;
	};
	
	double PassiveAgressiveRegression::predict(arma::dvec& data_point){
		return arma::dot(W,data_point);
	};
	
	double PassiveAgressiveRegression::RMSE(arma::mat& testbatch, arma::dvec& labels){
		
		// Calculate accuracy RMSE.
		double RMSE = 0;
		for(int i=0;i<labels.n_elem;i++){
			RMSE += std::pow( labels(i) - arma::dot(W,testbatch.unsafe_col(i)) , 2);
		}
		cout << endl << "(RMSE*T)^2 = " << RMSE << endl;
		RMSE /= labels.n_elem;
		RMSE = std::sqrt(RMSE);
		
		return RMSE;
		
	};
	
	class NN_Regressor{
	protected:
		string name = "Neural Network Regressor."; // The model type.
		Json::Value root; // JSON file to read the hyperparameters.  [optional]
		FFN< MeanSquaredError<> >* model; // The actual feed forward neurar network topology.
		
		double stepSize; // Learning rate of the optimizer.
		int batchSize; // The size of the batch that is to be fed to the Neural Network.
		double beta1; // The beta1 parameter of the Adam optimizer.
		double beta2; // The beta2 parameter of the Adam optimizer.
		double eps; // Epsilon of the optimizer.
		int maxIterations; // Maximum iterations of the optimizer.
		double tolerance; // Tolerance of the optimizer.
		
	public:
		NN_Regressor(const string cfg){
			
			try{
				
				std::ifstream cfgfile(cfg);
				cfgfile >> root;
				
				int sz_input_layer = root["NN_Classifier"]
							         .get("Size_of_input_layer",-1).asInt(); 
				if(sz_input_layer <= 0){
					cout << endl <<"Invalid size for input layers." << endl;
					cout << "Parameter Size_of_input_layer must be a positive integer." << endl;
					throw;
				}
				
				int num_hidden_lrs = root["NN_Classifier"]
							         .get("Number_of_hidden_layers",-1).asInt();
				if(num_hidden_lrs <= 0){
					cout << endl <<"Invalid parameter number of hidden layers." << endl;
					cout << "Parameter Number_of_hidden_layers must be a positive integer." << endl;
					throw;
				}
				
				stepSize = root["NN_Classifier"]
						   .get("stepSize",-1).asDouble();
				if(stepSize <= 0.){
					cout << endl <<"Invalid parameter stepSize given." << endl;
					cout << "Parameter stepSize must be a positive double." << endl;
					throw;
				}
				
				batchSize = root["NN_Classifier"]
							.get("batchSize",-1).asInt();
				if(batchSize <= 0 && batchSize >=5000){
					cout << endl <<"Invalid parameter batchSize given." << endl;
					cout << "Parameter batchSize must be a positive integer." << endl;
					throw;
				}
				
				beta1 = root["NN_Classifier"]
						.get("beta1",-1).asDouble();
				if(beta1 <= 0.){
					cout << endl <<"Invalid parameter beta1 given." << endl;
					cout << "Parameter beta1 must be a positive double." << endl;
					throw;
				}
				
				beta2 = root["NN_Classifier"]
						.get("beta2",-1).asDouble();
				if(beta2 <= 0.){
					cout << endl <<"Invalid parameter beta2 given." << endl;
					cout << "Parameter beta2 must be a positive double." << endl;
					throw;
				}
				
				eps = root["NN_Classifier"]
					  .get("eps",-1).asDouble();
				if(eps <= 0.){
					cout << endl <<"Invalid parameter eps given." << endl;
					cout << "Parameter eps must be a positive double." << endl;
					throw;
				}
				
				maxIterations = root["NN_Classifier"]
							    .get("maxIterations",-1).asInt();
				if(maxIterations <= 0 ){
					cout << endl <<"Invalid parameter maxIterations given." << endl;
					cout << "Parameter maxIterations must be a positive integer." << endl;
					throw;
				}
				
				tolerance = root["NN_Classifier"]
					        .get("tolerance",-1).asDouble();
				if(tolerance <= 0.){
					cout << endl <<"Invalid parameter tolerance given." << endl;
					cout << "Parameter tolerance must be a positive double." << endl;
					throw;
				}
				
				vector<int> layer_size;
				vector<string> layer_activation;
				for(unsigned i = 0; i < num_hidden_lrs; ++i){
					
					layer_size.push_back(root["NN_Classifier"]
					                     .get("hidden"+std::to_string(i+1),-1).asInt());
					if(layer_size.back()<=0){
						cout << endl << "Invalid size of hidden layer " << i+1 << "." << endl;
						cout << "Size of hidden layer must be a positive integer." << endl;
						throw;
					}
					
					layer_activation.push_back(root["NN_Classifier"]
					                           .get("hidden"+std::to_string(i+1)+"_Activation",-1).asString());
					if(layer_activation.back() != "logistic" &&
					   layer_activation.back() != "relu" &&
					   layer_activation.back() != "softplus" &&
					   layer_activation.back() != "softsign" &&
					   layer_activation.back() != "swish" &&
					   layer_activation.back() != "tanh"){
						cout << endl << "Invalid activation function for layer " << i+1 << "." << endl;
						cout << 
							 "Valid activation functions are : ['identity', 'logistic', 'relu', 'softplus', 'softsign', 'swish', 'tanh']." 
							 << endl;
						throw;
					}
				}
				
				// Built the model
				model = new FFN<MeanSquaredError<> >();
				for(unsigned i=0;i<num_hidden_lrs;i++){
					if(i==0){
						model->Add<Linear<> >(sz_input_layer,layer_size.at(i));
					}else{
						model->Add<Linear<> >(layer_size.at(i-1),layer_size.at(i));
					}
					if(layer_activation.at(i) == "logistic"){
						model->Add<SigmoidLayer<> >();
					}else if(layer_activation.at(i) == "tanh"){
						model->Add<TanHLayer<> >();
					}else if(layer_activation.at(i) == "relu"){
						model->Add<ReLULayer<> >();
					}else{
						cout << endl << "No support for this activation function at the moment." << endl;
						throw;
					}
				}
				model->Add<Linear<> >(layer_size.at(num_hidden_lrs-1), 1);
				
			}catch(...){
				throw;
			}
		
		};
		
		// Get the type of the regressor.
		string getName(){return name;};
		
		// Stream update
		void fit(arma::mat& batch, arma::mat& labels);
		
		// Make prediction on a batch.
		arma::mat predictOnBatch(arma::mat& batch);
		
		// Make a prediction on a data point.
		double predict(arma::mat& data_point);
		
		// Get score
		double RMSE(arma::mat& test_data, arma::mat& labels);
		
	};
	
	void NN_Regressor::fit(arma::mat& batch, arma::mat& labels){
		
		int mod = batch.n_cols % batchSize;
		int num_of_batches = std::floor( batch.n_cols / batchSize );
		
		cout << "batch.n_cols : " << batch.n_cols << endl;
		cout << "num_of_batches : " << num_of_batches << endl;
		cout << "mod : " << mod << endl;
		cout << arma::size(batch) << endl;
		cout << arma::size(labels) << endl;
		
		//SGD opt(0.001, 1, 1);
		//SGD opt(0.001,batch.n_cols,10,1e-5);
		
		if( num_of_batches > 1 ){
			
			AdamType opt( stepSize, batchSize, beta1, beta2, eps, maxIterations, tolerance );
			for(unsigned i = 0; i < num_of_batches; ++i){
				model->Train( batch.submat( 0, i*batchSize, batch.n_rows - 1, i*batchSize + batchSize - 1 ),
							  labels.submat( 0, i*batchSize, labels.n_rows - 1, i*batchSize + batchSize - 1 ),
							  opt );
			}
			if( mod > 0 ){
				model->Train( batch.submat( 0, batch.n_cols - mod, batch.n_rows - 1, batch.n_cols - 1 ),
							  labels.submat( 0, batch.n_cols - mod, labels.n_rows - 1, batch.n_cols - 1 ),
							  opt );
			}
			
		}else{
			
			AdamType opt( stepSize, batch.n_cols, beta1, beta2, eps, maxIterations, tolerance );
			model->Train( batch, labels, opt );
			
		}
		
	};
	
	arma::mat NN_Regressor::predictOnBatch(arma::mat& batch){
		arma::mat prediction;
		model->Predict(batch, prediction);
		return prediction;
	};
	
	double NN_Regressor::predict(arma::mat& data_point){
		
		// Check for invalid data point given
		if( data_point.n_cols < 0 || data_point.n_cols > 1){
			return -1;
		}
		
		arma::mat prediction;
		model->Predict(data_point, prediction);
		
		return arma::as_scalar(prediction);
		
	};
	
	double NN_Regressor::RMSE(arma::mat& test_data, arma::mat& labels){
		
		// Make predictions on the test dataset.
		arma::mat prediction;
		model->Predict(test_data, prediction);
		
		// Calculate accuracy RMSE.
		double RMSE = 0;
		for(int i=0;i<labels.n_elem;i++){
			RMSE += std::pow( labels(0,i) - prediction(0,i) , 2);
		}
		cout << endl << "(RMSE*T)^2 = " << RMSE << endl;
		RMSE /= labels.n_elem;
		RMSE = std::sqrt(RMSE);
		
		return RMSE;
		
	};
	
}
#endif