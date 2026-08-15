#include "Test.h"



void Tests::CheckResult(const std::vector<double>& res, int index) {
	int size = res.size();
	double middleRes = 0.0;
	for (const auto& n : res) {
		middleRes += n;
	}
	middleRes /= (double)size;
	if (fabs(testSetTargets[index] - middleRes) < PermissebleError) {
		correctAnswer++;
	}
}

unsigned int Tests::GetTestSetSize() const {
	return testSetSize;
}

int Tests::GetSuccessProcent() const {
	return (correctAnswer / testSetSize) * 100;
}

std::vector<std::string>  Tests::GetTestLog() const {
	return  Tests::testSetLogs;
}

void Tests::Test(const unsigned int& inputNeuronsAmount, const unsigned int& hidenNeuronsAmount, const unsigned int& outputNeuronsAmount) {
	Perceptron p(inputNeuronsAmount, hidenNeuronsAmount, outputNeuronsAmount);
		std::vector<std::string> log = GetTestLog();
		for (int i = 0; i < testSetSize; i++) {
			std::cout << "Log: " << log[i] << std::endl;
			p.ProccedString(log[i]);
			p.InputToHiddenLayerProccess();
			p.HiddenToOutputLayerProccess();
			std::vector<double> res = p.GetOutputLayer();
			std::cout << "Result: " << res[0] << std::endl;
			CheckResult(p.GetOutputLayer(), i);
			p.CleanInputLayer();
		}
	std::cout << "Proccent of corretc answers:\t" << GetSuccessProcent() << "%" << std::endl;
}