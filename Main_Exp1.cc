#include <iostream>
#include <string>
#include <vector>
#include <string>

#include "feeders.hh"

using namespace feeders;
using namespace feeders::MLPACK_feeders;
using namespace feeders::DLIB_feeders;
int main(int argc, char** argv){
	
	std::string cfg = std::string(argv[1]);
	
	//Random_Feeder<> simulation(cfg);
	//HDF5_Feeder<> simulation(cfg);
	LeNet_Feeder<unsigned char, unsigned long, ml_gm_proto::DL_gm_networks::GM_Net> simulation(cfg);
	
	simulation.initializeSimulation();
	simulation.printStarNets();
	simulation.TrainNetworks();
	
	return 0;
}
