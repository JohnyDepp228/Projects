#ifndef DATASETGENERATOR_H
#define DATASETGENERATOR_H
#include <vector>
#include <random>

struct Dataset {
	std::vector<int> dangerLevel;
	std::vector<std::string> logs;
	int testListSize;
	std::vector<std::string> methods;
	std::vector<std::string> pages;
	std::vector<std::string> protocol;
	std::vector<std::string> agent;


	std::vector<std::string> SQL;
	std::vector<std::string> XSS;
	std::vector<std::string> Win;

	Dataset(int testListSize) {
		this->testListSize = testListSize;
		dangerLevel.resize(testListSize);
		InitLogsDB();
	}
	void InitLogsDB();
	double Index(int leftBoard, int rightBoard);
	std::string CreateLog(const std::vector<std::string>& methods, const std::vector<std::string>& pages, const std::vector<std::string>& protocol,
		const std::vector<std::string>& agent,
		const std::vector<std::string>& SQL, const std::vector<std::string>& XSS, const std::vector<std::string>& Win,const int &danger);

	bool DangerLog();

	std::vector<std::string> GetLogs() const {
		return logs;
	}

	std::vector<int> GetLogsDanger() const {
		return dangerLevel;
	}

};



#endif DATASETGENERATOR_H
