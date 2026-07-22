#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <map>
#include <set>
#include "DatasetGenerator.h"

enum Errors {
	FILEREAD = 1,
	FILEREADVEC1D,
	FILEREADVEC2D,
	FILEWRITEVEC2D,
	FILEWRITEVEC1D,
	FILEWRITE,
	DIVISIONBYZERO
};

struct SavePerceprtonConfig {
	std::vector<std::vector<double>> InputToHiddenWeights;
	std::vector<std::vector<double>> hidenToOutputWeights;
	std::vector<std::vector<double>> hidenToOutputLayerVelocity;
	std::vector<std::vector<double>> inputToHiddenLayerVelocity;
	std::vector<double> inputToHiddenLayerBias;
	std::vector<double> hidenToOutputLayerBias;
	unsigned int inputNeuronsAmount;
	unsigned int hidenNeuronsAmount;
	unsigned int outputNeuronsAmount;

	SavePerceprtonConfig(std::vector<std::vector<double>> InputToHiddenWeights, std::vector<std::vector<double>> hidenToOutputWeights,
		std::vector<std::vector<double>> hidenToOutputLayerVelocity, std::vector<std::vector<double>> inputToHiddenLayerVelocity,
		std::vector<double> inputToHiddenLayerBias, std::vector<double> hidenToOutputLayerBias,
		const unsigned int& inputNeuronsAmount, const unsigned int& hidenNeuronsAmount, const unsigned int& outputNeuronsAmount) {
		this->InputToHiddenWeights = InputToHiddenWeights;
		this->hidenToOutputWeights = hidenToOutputWeights;
		this->hidenToOutputLayerVelocity = hidenToOutputLayerVelocity;
		this->inputToHiddenLayerVelocity = inputToHiddenLayerVelocity;
		this->inputToHiddenLayerBias = inputToHiddenLayerBias;
		this->hidenToOutputLayerBias = hidenToOutputLayerBias;
		this->inputNeuronsAmount = inputNeuronsAmount;
		this->hidenNeuronsAmount = hidenNeuronsAmount;
		this->outputNeuronsAmount = outputNeuronsAmount;
	}
	SavePerceprtonConfig() {
		this->InputToHiddenWeights.clear();
		this->hidenToOutputWeights.clear();
		this->hidenToOutputLayerVelocity.clear();
		this->inputToHiddenLayerVelocity.clear();
		this->inputToHiddenLayerBias.clear();
		this->hidenToOutputLayerBias.clear();

		this->inputNeuronsAmount = 0;
		this->hidenNeuronsAmount = 0;
		this->outputNeuronsAmount = 0;
	}

	void SaveToFile(std::string path, const std::vector<std::vector<double>>& InputToHiddenWeights, const std::vector<std::vector<double>>& hidenToOutputWeights,
		const std::vector<std::vector<double>>& hidenToOutputLayerVelocity, const std::vector<std::vector<double>>& inputToHiddenLayerVelocity,
		const std::vector<double>& inputToHiddenLayerBias, const std::vector<double>& hidenToOutputLayerBias,
		const unsigned int& inputNeuronsAmount, const unsigned int& hidenNeuronsAmount, const unsigned int& outputNeuronsAmount) {
		SavePerceprtonConfig config(InputToHiddenWeights, hidenToOutputWeights, hidenToOutputLayerVelocity, inputToHiddenLayerVelocity, inputToHiddenLayerBias, hidenToOutputLayerBias,
			inputNeuronsAmount, hidenNeuronsAmount, outputNeuronsAmount);
		std::ofstream fout;
		fout.open(path, std::ios::binary);
		auto SaveVec2D = [&fout](const std::vector<std::vector<double>>& vec) {
			size_t size = vec.size();
			fout.write((char*)&size, sizeof(size_t));
			for (auto row : vec) {
				size_t rowSize = row.size();
				if (!fout.write((char*)&rowSize, sizeof(size_t))) {
					throw Errors::FILEWRITEVEC2D;
				}
				if (rowSize > 0) {
					if (!fout.write((char*)row.data(), sizeof(double) * rowSize)) {
						throw Errors::FILEWRITEVEC2D;
					}
				}
			}
			};

		auto SaveVec1D = [&fout](const std::vector<double>& vec) {
			size_t rowSize = vec.size();
			if (!fout.write((char*)&rowSize, sizeof(size_t))) {
				throw Errors::FILEWRITEVEC1D;
			}
			if (rowSize > 0) {
				if (!fout.write((char*)vec.data(), sizeof(double) * rowSize)) {
					throw Errors::FILEWRITEVEC1D;
				}
			}
			};

		SaveVec2D(config.InputToHiddenWeights);
		SaveVec2D(config.hidenToOutputWeights);
		SaveVec2D(config.hidenToOutputLayerVelocity);
		SaveVec2D(config.inputToHiddenLayerVelocity);
		SaveVec1D(config.inputToHiddenLayerBias);
		SaveVec1D(config.hidenToOutputLayerBias);
		if (!fout.write((char*)&inputNeuronsAmount, sizeof(unsigned int))) {
			throw Errors::FILEWRITE;
		}
		if (!fout.write((char*)&hidenNeuronsAmount, sizeof(unsigned int))) {
			throw Errors::FILEWRITE;
		}
		if (!fout.write((char*)&outputNeuronsAmount, sizeof(unsigned int))) {
			throw Errors::FILEWRITE;
		}
		fout.close();
	}

	bool GetFromFile(std::string path) {
		SavePerceprtonConfig config;
		std::ifstream fin;
		fin.open(path, std::ios::binary);

		auto SetVec2D = [&fin](std::vector<std::vector<double>>& vec) {
			size_t size = 0;
			fin.read((char*)&size, sizeof(size_t));
			vec.resize(size);
			for (auto& row : vec) {
				size_t rowSize;
				if (!fin.read((char*)&rowSize, sizeof(size_t))) {
					throw Errors::FILEREADVEC2D;
				}
				row.resize(rowSize);
				if (rowSize > 0) {
					if (!fin.read((char*)row.data(), sizeof(double) * rowSize)) {
						throw Errors::FILEREADVEC2D;
					}
				}
			}
			};

		auto SetVec1D = [&fin](std::vector<double>& vec) {
			size_t rowSize = 0;
			if (!fin.read((char*)&rowSize, sizeof(size_t))) {
				throw Errors::FILEREADVEC1D;
			}
			vec.resize(rowSize);
			if (rowSize > 0) {
				if (!fin.read((char*)vec.data(), sizeof(double) * rowSize)) {
					throw Errors::FILEREADVEC1D;
				}
			}
			};
		if (fin.is_open()) {
			SetVec2D(config.InputToHiddenWeights);
			SetVec2D(config.hidenToOutputWeights);
			SetVec2D(config.hidenToOutputLayerVelocity);
			SetVec2D(config.inputToHiddenLayerVelocity);
			SetVec1D(config.inputToHiddenLayerBias);
			SetVec1D(config.hidenToOutputLayerBias);
			if (!fin.read((char*)&config.inputNeuronsAmount, sizeof(unsigned int))) {
				throw Errors::FILEREAD;
			}
			if (!fin.read((char*)&config.hidenNeuronsAmount, sizeof(unsigned int))) {
				throw Errors::FILEREAD;
			}
			if (!fin.read((char*)&config.outputNeuronsAmount, sizeof(unsigned int))) {
				throw Errors::FILEREAD;
			}
			SetConfig(config);
			fin.close();
			return true;
		}
		else {
			fin.close();
			return false;
		}
	}

	void SetConfig(std::vector<std::vector<double>>& InputToHiddenWeights, std::vector<std::vector<double>>& hidenToOutputWeights,
		std::vector<std::vector<double>>& hidenToOutputLayerVelocity, std::vector<std::vector<double>>& inputToHiddenLayerVelocity,
		std::vector<double>& inputToHiddenLayerBias, std::vector<double>& hidenToOutputLayerBias,
		unsigned int& inputNeuronsAmount, unsigned int& hidenNeuronsAmount, unsigned int& outputNeuronsAmount)
	{
		InputToHiddenWeights = this->InputToHiddenWeights;
		hidenToOutputWeights = this->hidenToOutputWeights;
		hidenToOutputLayerVelocity = this->hidenToOutputLayerVelocity;
		inputToHiddenLayerVelocity = this->inputToHiddenLayerVelocity;
		inputToHiddenLayerBias = this->inputToHiddenLayerBias;
		hidenToOutputLayerBias = this->hidenToOutputLayerBias;
		inputNeuronsAmount = this->inputNeuronsAmount;
		hidenNeuronsAmount = this->hidenNeuronsAmount;
		outputNeuronsAmount = this->outputNeuronsAmount;

	}

	void SetConfig(const SavePerceprtonConfig& config)
	{
		this->InputToHiddenWeights = config.InputToHiddenWeights;
		this->hidenToOutputWeights = config.hidenToOutputWeights;
		this->hidenToOutputLayerVelocity = config.hidenToOutputLayerVelocity;
		this->inputToHiddenLayerVelocity = config.inputToHiddenLayerVelocity;
		this->inputToHiddenLayerBias = config.inputToHiddenLayerBias;
		this->hidenToOutputLayerBias = config.hidenToOutputLayerBias;
		this->inputNeuronsAmount = config.inputNeuronsAmount;
		this->hidenNeuronsAmount = config.hidenNeuronsAmount;
		this->outputNeuronsAmount = config.outputNeuronsAmount;

	}
};

class Perceptron {
private:

	std::vector<std::vector<double>> InputToHiddenWeights;//save
	std::vector<std::vector<double>> hidenToOutputWeights;//save
	std::vector<std::vector<double>> hidenToOutputLayerVelocity;//save
	std::vector<std::vector<double>> inputToHiddenLayerVelocity;//save

	std::vector<double> inputLayer;
	std::vector<double> hidenLayer;
	std::vector<double> hidenLayerBeforeReLu;
	std::vector<double> outputLayer;

	std::vector<double> inputToHiddenLayerBias;//save
	std::vector<double> hidenToOutputLayerBias;//save

	std::vector<double> inputToHidenError;
	std::vector<double> hidenToOutputError;


	std::vector<int> learningTargets;
	std::vector<std::string> learningLogs;

	double LR = 0.01;
	double inertia = 0.1;

	unsigned int inputNeuronsAmount;
	unsigned int hidenNeuronsAmount;
	unsigned int outputNeuronsAmount;
	unsigned int epoch;

public:
	Perceptron(const unsigned int& inputNeuronsAmount, const unsigned int& hidenNeuronsAmount, const unsigned int& outputNeuronsAmount,
		const std::vector<int>& targets, const std::vector<std::string>& learningLogs) {
		SavePerceprtonConfig config;
		epoch = 0;
		if (config.GetFromFile("C:/Users/LordMegatron/Desktop/Pupa/weights.txt")) {
			config.SetConfig
			(
				this->InputToHiddenWeights, this->hidenToOutputWeights,
				this->hidenToOutputLayerVelocity, this->inputToHiddenLayerVelocity,
				this->inputToHiddenLayerBias, this->hidenToOutputLayerBias,
				this->inputNeuronsAmount, this->hidenNeuronsAmount, this->outputNeuronsAmount
			);
			this->learningTargets = targets;
			this->learningLogs = learningLogs;
			inputLayer = std::vector<double>(this->inputNeuronsAmount, 0);
			hidenLayer = std::vector<double>(this->hidenNeuronsAmount, 0);
			hidenLayerBeforeReLu = std::vector<double>(this->hidenNeuronsAmount, 0);
			outputLayer = std::vector<double>(this->outputNeuronsAmount, 0);




		}
		else {
			this->inputNeuronsAmount = inputNeuronsAmount;
			this->hidenNeuronsAmount = hidenNeuronsAmount;
			this->outputNeuronsAmount = outputNeuronsAmount;
			this->learningTargets = targets;
			this->learningLogs = learningLogs;

			inputLayer = std::vector<double>(inputNeuronsAmount, 0);
			hidenLayer = std::vector<double>(hidenNeuronsAmount, 0);
			hidenLayerBeforeReLu = std::vector<double>(hidenNeuronsAmount, 0);
			outputLayer = std::vector<double>(outputNeuronsAmount, 0);

			inputToHiddenLayerBias = std::vector<double>(hidenNeuronsAmount, 0.01);
			hidenToOutputLayerBias = std::vector<double>(outputNeuronsAmount, 0);

			inputToHidenError = std::vector<double>(hidenNeuronsAmount, 0.0);
			hidenToOutputError = std::vector<double>(outputNeuronsAmount, 0.0);

			hidenToOutputLayerVelocity = std::vector<std::vector<double>>(hidenNeuronsAmount, std::vector<double>(outputNeuronsAmount, 0));
			inputToHiddenLayerVelocity = std::vector<std::vector<double>>(inputNeuronsAmount, std::vector<double>(hidenNeuronsAmount, 0));

			InputToHiddenWeights = std::vector<std::vector<double>>(inputNeuronsAmount, std::vector<double>(hidenNeuronsAmount, 0));
			hidenToOutputWeights = std::vector<std::vector<double>>(hidenNeuronsAmount, std::vector<double>(outputNeuronsAmount, 0));

			InitWeights(InputToHiddenWeights);
			InitWeights(hidenToOutputWeights);



			Learning();
			config.SaveToFile("C:/Users/LordMegatron/Desktop/Pupa/weights.txt", this->InputToHiddenWeights, this->hidenToOutputWeights,
				this->hidenToOutputLayerVelocity, this->inputToHiddenLayerVelocity,
				this->inputToHiddenLayerBias, this->hidenToOutputLayerBias,
				this->inputNeuronsAmount, this->hidenNeuronsAmount, this->outputNeuronsAmount);
		}
	}

	double ReLu(const double& data) {
		return data > 0 ? data : 0;
	}

	double directiveReLu(const double& data) {
		return data > 0 ? 1.0 : 0;
	}

	double Sigmoid(const double& data) {
		return 1.0 / (1.0 + std::exp(-data));
	}

	double directiveSigmoid(const double& data) {
		return data * (1.0 - data); // data after sigmoid
	}

	void SetInputLayer(const std::vector<double>& input) {
		std::fill(inputLayer.begin(), inputLayer.end(), 0.0);
		if (input.size() == inputLayer.size()) {
			inputLayer = input;
		}
		else {
			inputLayer.assign(input.begin(), input.begin() + inputLayer.size());
		}
	}

	void InputToHiddenLayerProccess() {
		std::fill(hidenLayer.begin(), hidenLayer.end(), 0.0);
		for (int i = 0; i < inputNeuronsAmount; i++) {
			for (int j = 0; j < hidenNeuronsAmount; j++) {
				hidenLayer[j] += (inputLayer[i] * InputToHiddenWeights[i][j]);
			}
		}

		for (int j = 0; j < hidenNeuronsAmount; j++) {
			hidenLayerBeforeReLu[j] = hidenLayer[j] + inputToHiddenLayerBias[j];
			hidenLayer[j] = ReLu(hidenLayer[j] + inputToHiddenLayerBias[j]);

		}
	}

	void HiddenToOutputLayerProccess() {
		std::fill(outputLayer.begin(), outputLayer.end(), 0.0);
		for (int i = 0; i < hidenNeuronsAmount; i++) {
			for (int j = 0; j < outputNeuronsAmount; j++) {
				outputLayer[j] += (hidenLayer[i] * hidenToOutputWeights[i][j]);
			}
		}
		for (int j = 0; j < outputNeuronsAmount; j++) {
			outputLayer[j] = Sigmoid(outputLayer[j] + hidenToOutputLayerBias[j]);

		}
	}

	std::vector<double> GetOutputLayer() const {
		return outputLayer;
	}

	void CleanInputLayer() {
		std::fill(inputLayer.begin(), inputLayer.end(), 0.0);
	}

	void InitWeights(std::vector<std::vector<double>>& layer) {
		for (auto& colls : layer) {
			for (auto& rows : colls) {
				rows = Weight(-1.0, 1.0);
			}
		}
	}


	double Weight(double leftBoard, double rightBoard) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(leftBoard, rightBoard);
		return dis(gen);
	}

	int hash(std::string key) {
		std::hash<std::string> hasher;
		return hasher(key) % inputNeuronsAmount;
	}

	void ProccedString(std::string str) {
		if (str.empty()) return;
		std::vector<std::string> words;
		std::string current_word = "";

		for (char c : str) {
			if (c == ' ') {
				if (!current_word.empty()) {
					words.push_back(current_word);
					current_word = "";
				}
			}
			else {
				current_word.push_back(c);
			}
		}
		if (!current_word.empty()) {
			words.push_back(current_word);
		}

		for (const auto& word : words) {
			int idx = hash(word);
			if (idx < inputLayer.size()) {
				inputLayer[idx] = 1.0;
			}
		}
	}



	double NewVelocity(const double& neuronError, const double& neuronInput, double& oldVelocity) {
		return oldVelocity = (inertia * oldVelocity) + (LR * neuronError * neuronInput);
	}

	void UpdateBias() {
		for (int i = 0; i < inputToHiddenLayerBias.size(); i++) {
			inputToHiddenLayerBias[i] = inputToHiddenLayerBias[i] - LR * inputToHidenError[i];
		}

		for (int i = 0; i < hidenToOutputLayerBias.size(); i++) {
			hidenToOutputLayerBias[i] = hidenToOutputLayerBias[i] - LR * hidenToOutputError[i];
		}
	}

	void hidenToOutError() {
		for (int i = 0; i < outputNeuronsAmount; i++) {
			hidenToOutputError[i] = (outputLayer[i] - learningTargets[epoch % learningTargets.size()]) * directiveSigmoid(outputLayer[i]);
		}
	}

	void inToHidenError() {
		for (int i = 0; i < hidenNeuronsAmount; i++) {
			double errorSum = 0;
			for (int j = 0; j < outputNeuronsAmount; j++) {
				errorSum += hidenToOutputError[j] * hidenToOutputWeights[i][j];
			}
			inputToHidenError[i] = errorSum * directiveReLu(hidenLayerBeforeReLu[i]);
		}
	}

	double SumVec(std::vector<double> vec) {
		double sum = 0;
		for (auto i : vec) {
			sum += i;
		}

		return sum;
	}

	void UpdateWeights() {
		for (int i = 0; i < hidenNeuronsAmount; i++) {
			for (int j = 0; j < outputNeuronsAmount; j++) {
				hidenToOutputWeights[i][j] -= NewVelocity(hidenToOutputError[j], hidenLayer[i], hidenToOutputLayerVelocity[i][j]);

			}
		}

		for (int i = 0; i < inputNeuronsAmount; i++) {
			for (int j = 0; j < hidenNeuronsAmount; j++) {
				InputToHiddenWeights[i][j] -= NewVelocity(inputToHidenError[j], inputLayer[i], inputToHiddenLayerVelocity[i][j]);
			}
		}

	}

	double MSE() {
		double errorSum = 0;
		for (int i = 0; i < outputNeuronsAmount; i++) {
			if (learningTargets.size() <= 0) {
				throw Errors::DIVISIONBYZERO;
			}
			errorSum += std::pow((outputLayer[i] - learningTargets[epoch % learningTargets.size()]), 2);
		}

		return (1.0 / outputNeuronsAmount) * errorSum;
	}

	double RMSE() {
		return std::sqrt(MSE());
	}

	void Learning() {
		std::vector<double> res(1, 0);
		int i = 0;
		while (i < learningTargets.size() - 1) {
			epoch = i;
			ProccedString(learningLogs[i]);
			std::cout << "Learning on log:\t" << learningLogs[i] << "\tDanger:\t" << learningTargets[i] << std::endl;
			InputToHiddenLayerProccess();
			HiddenToOutputLayerProccess();
			res = GetOutputLayer();
			for (auto n : res) {
				std::cout << "Result: " << n << "\tError: " << RMSE() << std::endl;
			}
			hidenToOutError();
			inToHidenError();
			UpdateBias();
			UpdateWeights();
			CleanInputLayer();
			i++;
			if ((RMSE() * 100) < 2) {
				break;
			}
		}
		std::cout << "Learning done on epoch " << epoch << std::endl;
	}

};

int main() {
	setlocale(LC_ALL, "ru");
	Dataset dataset(200);
	std::vector<std::string> log = dataset.GetLogs();
	std::vector<int> danger = dataset.GetLogsDanger();
	std::vector<double> res(1, 0);
	Perceptron p(500, 128, 1, danger, log);
	try {
		std::cout << "Log:\t" << log[3] << std::endl;
		p.ProccedString(log[3]);
		p.InputToHiddenLayerProccess();
		p.HiddenToOutputLayerProccess();
		res = p.GetOutputLayer();
		for (auto n : res) {
			std::cout << "Result: " << n << "\tError: " << p.RMSE() << std::endl;
		}
	}
	catch (const Errors& e) {
		if (e == Errors::FILEREAD || e == FILEREADVEC1D || e == FILEREADVEC2D) {
			std::cout << "Can't read perceptron weights. Error code\t" << e << std::endl;
		}
		else if (e == Errors::FILEWRITE || e == Errors::FILEWRITEVEC1D || e == Errors::FILEWRITEVEC2D) {
			std::cout << "Can't save perceptron weights. Errro code\t" << e << std::endl;
		}
		else if (e == Errors::DIVISIONBYZERO) {
			std::cout << "DIVISION BY ZERO. Error code\t" << e << std::endl;
		}
	}
	catch (...) {
		std::cout << "Unknown problem" << std::endl;
	}
	return 0;
}