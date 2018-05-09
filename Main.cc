#include <iostream>
#include <string>
#include <vector>
#include <string>

#include "feeders.hh"

//const std::string cfg = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/inputs.json";
const std::string cfg = "/home/aris/Desktop/Diplwmatikh/Starting_Cpp_Developing/DLIB_inputs.json";

using namespace feeders;
using namespace feeders::MLPACK_feeders;
using namespace feeders::DLIB_feeders;
int main(void){
	
	//HDF5_Feeder<double> simulation(cfg);
	//Random_Feeder simulation(cfg);
	LeNet_Feeder<unsigned char, unsigned long> simulation(cfg);
	
	simulation.initializeSimulation();
	simulation.printStarNets();
	simulation.TrainNetworks();
	
	return 0;
}
