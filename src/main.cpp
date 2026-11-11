#include <chrono>
#include "config.h"
#include "convolutionallayer.h"
#include <iostream>
#include "classifier.h"
#include "readandshuffle.h"

int main(){
ReadAndShuffle rs;
std::string datasetPathTrain = "../Pupa/dataset/fish-vista_split/train";
std::string datasetPathTest  = "../Pupa/dataset/fish-vista_split/test";
std::vector<DatasetItem> train;
std::vector<DatasetItem> test;
rs.ReadAndShuffleTrain(train,datasetPathTrain);
rs.ReadAndShuffleTrain(test,datasetPathTest);

ConvolutionalLayer convLay;
Classifier cl;
convLay.SetChannalsNum(6);
convLay.SetChannalsSize(16);
auto start = std::chrono::high_resolution_clock::now();

try{
	int epoch = 0;
	convLay.SetChannelsParams();
	while(epoch < 5){
		double error = 0.0;
		for(int i =0; i < 10; i++){
			convLay.ClearChannelsMemory(); 
			cl.Reset();
			convLay.SetImage(train[i].path);
			std::vector<double> forwardRes  = convLay.Forward();
			cl.SetFullyConnectedLayer(forwardRes);
			cl.Classification();
			std::vector<double> out = cl.GetOutLayer();
		  	cl.LearningClassifier(train[i].targetAnswer);
			std::cout << "Img " << i << " | Target: " << train[i].targetAnswer 
          	<< "| Answer: " << cl.FindCorrectOutNeuro() << " | Error: " << cl.GetError() << "\n";
			convLay.Backpropageation(cl.GetInputErrors());
			error += cl.GetError();
		}
		std::cout << "Epoch\t" << epoch << "\tError:\t" << error << std::endl;
		epoch++;
	}
}
	catch (const std::string& msg) {
        std::cout << "Caught string exception: " << msg << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "Caught std::exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cout << "Unknown crash occurred!" << std::endl;
    }

// Config conf;
// try{
// 	convLay.SetConfigPath("./config.txt");
// 	convLay.SaveChannelConfig();
// 	convLay.ReadChannelConfig();
// }
// catch(std::string &msg){
// 	std::cout << msg << std::endl;
// }


auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = end - start;
std::cout << "\nExecution time: " << duration.count() << std::endl;

return 0;
}