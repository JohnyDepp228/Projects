#include "Test.h"



void Tests::Testing(const std::vector<double>& res, int index) {
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
