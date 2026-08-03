#include "perceptron.h"


int main() {
	
	std::string str;

	Perceptron p(100, 80, 1);

	p.Test();

	std::cout << "Enter log to test: " << std::endl;

	getline(std::cin, str);

	p.FullProcess(str);


	return 0;
}