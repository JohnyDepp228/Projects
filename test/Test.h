#ifndef TEST_H
#define TEST_H



#include "DatasetGenerator.h"
#include <vector>



class Tests {
private:
	unsigned int testSetSize = 1000;
	Dataset* testSet;
	std::vector<double> testSetTargets;
	std::vector<std::string> testSetLogs;
	int correctAnswer = 0;
	double PermissebleError = 0.2;
public:

	Tests() {
		testSet = new Dataset(testSetSize);
		testSetTargets = testSet->GetLogsDanger();
		testSetLogs = testSet->GetLogs();

	}
	void Testing(const std::vector<double>& res, int index);

	unsigned int GetTestSetSize() const;

	std::vector<std::string> GetTestLog() const;
	
	~Tests() {
		delete testSet;
	}

	int GetSuccessProcent() const;

};


#endif