#ifndef DATASETGENERATOR_H
#define DATASETGENERATOR_H
#include <vector>
#include <random>
#include <iostream>

struct Dataset {
	std::vector<double> map;
	int correctAnswer = 0;


	void SetAnswer(int answer) {
		this->correctAnswer = answer;
	}

	void SetMap(const std::vector<double> &map) {
		this->map = map;
	}

	int GetAnswer() const {
		return this->correctAnswer;
	}

	std::vector<double>  GetMap() const {
		return this->map;
	}

};


class GenerateDataset {
private:
	int datasetSize = 0;
	Dataset* dataset;

public:
	GenerateDataset(int size) {
		this->datasetSize = size;
		dataset = new Dataset[size];
        CreateDataset();
	}
    int GetWinner(const std::vector<double>& map);

    bool IsGameOver(const std::vector<double>& map);

    int GetRandomMove(const std::vector<double>& map);

    int Minimax(std::vector<double>& map, bool oTurn);

    int GetBestMove(std::vector<double>& map);

	void GenerateGame(int& datasetIndex);

	void CreateDataset();

	std::vector<double> GetMap(int &index) {

		return dataset[index].GetMap();
	}

    int GetAnswer(int& index) {

        return dataset[index].GetAnswer();
    }
	~GenerateDataset() {
		delete[] dataset;
	}
};

#endif 
