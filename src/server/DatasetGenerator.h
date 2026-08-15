#ifndef DATASETGENERATOR_H
#define DATASETGENERATOR_H
#include <vector>
#include <random>

struct Dataset {
	std::vector<double> dangerLevel;
	std::vector<std::string> logs;
	std::vector<std::string> methods;
	std::vector<std::string> pages;
	std::vector<std::string> protocol;
	std::vector<std::string> agent;


	std::vector<std::string> SQL;
	std::vector<std::string> XSS;
	std::vector<std::string> Win;

	int testListSize;
	//int uniqueWordsAmount = 37;

	Dataset(int testListSize) {
		this->testListSize = testListSize;
		dangerLevel.resize(testListSize);
		InitLogsDB();
	}
	Dataset() {
		this->testListSize = 0;
	}
	void InitLogsDB();
	int Index(int leftBoard, int rightBoard);
	std::string CreateLog(const std::vector<std::string>& methods, const std::vector<std::string>& pages, const std::vector<std::string>& protocol,
		const std::vector<std::string>& agent,
		const std::vector<std::string>& SQL, const std::vector<std::string>& XSS, const std::vector<std::string>& Win, const int& danger);

	bool DangerLog();

	std::vector<std::string> GetLogs() const {
		return logs;
	}

	std::vector<double> GetLogsDanger() const {
		return dangerLevel;
	}

	/*int GetNumOfUniqueWords() const {
		return uniqueWordsAmount;
	}*/

};

#endif 