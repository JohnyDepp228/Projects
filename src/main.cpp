#include <chrono>
#include "config.h"
#include "convolutionallayer.h"
#include <iostream>
#include "classifier.h"


int main(){
 
ConvolutionalLayer convLay;
Classifier cl;
convLay.SetChannalsNum(6);
convLay.SetChannalsSize(16);
auto start = std::chrono::high_resolution_clock::now();

// try{
// 	int epoch = 0;
// 	double error;
// 	std::string path = "C:/Users/LordMegatron/Desktop/2.jpg";
// 	//std::string path = "C:/Users/Boss/Desktop/2.jpg";
// 	int correctAnswer1 = 2;
// 	std::string path2 = "C:/Users/LordMegatron/Desktop/3.png";
// 	int correctAnswer2 = 3;
// 	std::cout << "Learning\n";
// 	convLay.SetChannelsParams();
// 	while(epoch < 20){
// 		std::cout << "Epoch:\t" << epoch << std::endl;
// 		convLay.SetImage(path);
// 		std::vector<double> forwardRes  = convLay.Forward();
// 		cl.SetFullyConnectedLayer(forwardRes);
// 		cl.Classification();
// 		cl.LearningClassifier(correctAnswer1);
// 		convLay.Backpropageation(cl.GetInputErrors());
// 		std::cout << "Error:\t" << cl.GetError() << std::endl;

		
// 		convLay.SetImage(path2);
// 		forwardRes = convLay.Forward();
// 		cl.SetFullyConnectedLayer(forwardRes);
// 		cl.Classification();
// 		cl.LearningClassifier(correctAnswer2);
// 		convLay.Backpropageation(cl.GetInputErrors());
// 		std::cout << "Error:\t" << cl.GetError() << std::endl;
// 		epoch++;
// 	}

// 	convLay.SetImage(path);
// 	std::cout << "Test 1" << std::endl;
// 	for(int i =0;i < 3;i++){
// 		std::vector<double> res = convLay.Forward();
// 		cl.SetFullyConnectedLayer(res);
// 		cl.Classification();
// 		if(cl.FindCorrectOutNeuro() == correctAnswer1){
// 			std::cout << "Passed test " << i << std::endl;
// 		}
// 		else {
// 			std::cout << "Failed test " << i << std::endl;
// 		}
// 	}

// 	convLay.SetImage(path2);
// 	std::cout << "Test 2" << std::endl;
// 	for(int i =0;i < 3;i++){
// 		std::vector<double> res = convLay.Forward();
// 		cl.SetFullyConnectedLayer(res);
// 		cl.Classification();
// 		if(cl.FindCorrectOutNeuro() == correctAnswer2){
// 			std::cout << "Passed test " << i << std::endl;
// 		}
// 		else {
// 			std::cout << "Failed test " << i << std::endl;
// 		}
// 	}
// }
// 	catch (const std::string& msg) {
//         std::cout << "Caught string exception: " << msg << std::endl;
//     }
//     catch (const std::exception& e) {
//         std::cout << "Caught std::exception: " << e.what() << std::endl;
//     }
//     catch (...) {
//         std::cout << "Unknown crash occurred!" << std::endl;
//     }

Config conf;
try{
	conf.SetPath("./conf.txt");
	conf.SaveData("size", 2);
	std::vector<double> temp(10,1.3);

	std::vector<double> temp2;
	conf.SaveData("weights", temp);
	int res =  -1;
	conf.ReadData("size",res);
	std::cout << res << std::endl;

	conf.ReadData("weights",temp2);
	for(auto n: temp2){
		std::cout << n << ' ';
	}
}
catch(std::string &msg){
	std::cout << msg << std::endl;
}


auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = end - start;
std::cout << "\nExecution time: " << duration.count() << std::endl;

return 0;
}