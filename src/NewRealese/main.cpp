
#include <random>
#include <chrono>
#include "convolutionallayer.h"


double Rand(){
    std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(0.0, 10.0);
		return dis(gen);
}

int main(){
    
ConvolutionalLayer convLay;
convLay.SetChannalsNum(6);
convLay.SetChannalsSize(16);
auto start = std::chrono::high_resolution_clock::now();

try{
	std::vector<double> res = convLay.Forward();
	std::cout << res.size();
}
catch(const char *msg){
	std::cout << msg;
}

auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = end - start;
std::cout << "Execution time: " << duration.count() << std::endl;

return 0;
}