#include <iostream>
#include <string>
#include <vector>
#include <string>

#include "feeders.hh"

//const std::string cfg = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/inputs.json";
//const std::string cfg = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/inputs2.json";
const std::string cfg = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/ML_inputs.json";
//const std::string cfg = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/DLIB_inputs.json";

using namespace feeders;
using namespace feeders::MLPACK_feeders;
using namespace feeders::DLIB_feeders;
int main(void){
	
	//Random_Feeder<> simulation(cfg);
	//HDF5_Feeder<> simulation(cfg);
	
	//HDF5_Drift_Feeder<ml_gm_proto::ML_gm_networks::GM_Net> simulation(cfg);
	HDF5_Drift_Feeder<ml_gm_proto::ML_fgm_networks::FGM_Net> simulation(cfg);
	
	//LeNet_Feeder<unsigned char, unsigned long, ml_gm_proto::DL_gm_networks::GM_Net> simulation(cfg);
	//LeNet_Feeder<unsigned char, unsigned long, ml_gm_proto::DL_fgm_networks::FGM_Net> simulation(cfg);
	
	simulation.initializeSimulation();
	simulation.printStarNets();
	simulation.TrainNetworks();
	
	return 0;
}
