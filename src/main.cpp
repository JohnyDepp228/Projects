
#include <random>
#include <chrono>
#include "convolutionallayer.h"
#include <iostream>
#include "classifier.h"
#include <float.h>



double Rand(){
    std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(0.0, 10.0);
		return dis(gen);
}

int main(){
 
ConvolutionalLayer convLay;
Classifier cl;
convLay.SetChannalsNum(6);
convLay.SetChannalsSize(16);
auto start = std::chrono::high_resolution_clock::now();

try{
	std::vector<double> res = convLay.Forward();
	cl.SetFullyConnectedLayer(res);
	cl.Classification();
	cl.LearningClassifier(2);
	std::cout << cl.GetInputErrors().size();
	convLay.Backpropageation(cl.GetInputErrors());

	std::cout << "Backprop done" << std::endl;

	std::cout << "Answer:" << cl.FindCorrectOutNeuro() << std::endl;
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

auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = end - start;
std::cout << "\nExecution time: " << duration.count() << std::endl;

return 0;
}