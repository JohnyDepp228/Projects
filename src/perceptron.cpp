#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <map>
#include <set>
#include <Windows.h>
#include <conio.h>
#include "DatasetGenerator.h"
#include "SentenceVectorize.h"
#include "Test.h"
#include "SaveConfig.h"
#include "Errors.h"

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


	std::vector<double> learningTargets;
	std::vector<std::string> learningLogs;

	double LR = 0.5;
	double inertia = 0.01;
	double oldWeight1;
	double oldWeight2;

	unsigned int inputNeuronsAmount;
	unsigned int hidenNeuronsAmount;
	unsigned int outputNeuronsAmount;
	unsigned int epoch;

	WordsVectorize *vectorize;

	Dataset *dataset;

public:
	Perceptron(const unsigned int &inputNeuronsAmount,const unsigned int &hidenNeuronsAmount,const unsigned int &outputNeuronsAmount) {
		SavePerceprtonConfig config;
		epoch = 0;
		dataset = new Dataset(2000);
		this->learningTargets = dataset->GetLogsDanger();
		this->learningLogs = dataset->GetLogs();
		unsigned int uniqueWordsInDataset = 37;
		vectorize = new WordsVectorize(learningLogs, dataset->methods, dataset->pages, dataset->protocol,
			dataset->agent, dataset->SQL, dataset->XSS, dataset->Win);
		if (config.GetFromFile("C:/Users/LordMegatron/Desktop/Pupa/weights.txt")) {
				config.SetConfig
				(
					this->InputToHiddenWeights, this->hidenToOutputWeights,
					this->hidenToOutputLayerVelocity, this->inputToHiddenLayerVelocity, 
					this->inputToHiddenLayerBias, this->hidenToOutputLayerBias,
					this->inputNeuronsAmount, this->hidenNeuronsAmount, this->outputNeuronsAmount
				);
				inputLayer = std::vector<double>(this->inputNeuronsAmount, 0);
				hidenLayer = std::vector<double>(this->hidenNeuronsAmount, 0);
				hidenLayerBeforeReLu = std::vector<double>(this->hidenNeuronsAmount, 0);
				outputLayer = std::vector<double>(this->outputNeuronsAmount, 0);
				vectorize->ReadIDFFromFile();	
		}
		else {
			this->inputNeuronsAmount = inputNeuronsAmount;
			this->hidenNeuronsAmount = hidenNeuronsAmount;
			this->outputNeuronsAmount = outputNeuronsAmount;

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

	double ReLu(const double &data) {
		//return data > 0 ? data : 0;
		return data > 0 ? data : data * 0.01; //leakyReLu
	}

	double directiveReLu(const double &data) {
		//return data > 0 ? 1.0 : 0;
		return data > 0 ? 1.0 : 0.01; //Leaky ReLu
	}

	double Sigmoid(const double& data) {
		return 1.0 / (1.0 + std::exp(-data));
	}

	double directiveSigmoid(const double& data) {
		return data * (1.0 - data); // data after sigmoid
	}

	void SetInputLayer(const std::vector<double>& input) {
		std::fill(inputLayer.begin(), inputLayer.end(), 0.0);
		if (input.size() <= inputLayer.size()) {
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
			hidenLayerBeforeReLu[j] = 0.0;
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
				rows = Weight(-0.5, 0.5);
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
		SetInputLayer(  vectorize->TFxIDF(str,inputNeuronsAmount));
	}

	double NewVelocity(double LR,const double& neuronError, const double& neuronInput,double &oldVelocity) {
		return oldVelocity = (inertia * oldVelocity) + (LR * neuronError * neuronInput);
	}

	void UpdateBias() {
		for (int i = 0; i < inputToHiddenLayerBias.size(); i++) {
			inputToHiddenLayerBias[i] += LR * inputToHidenError[i];
		}

		for (int i = 0; i < hidenToOutputLayerBias.size(); i++) {
			hidenToOutputLayerBias[i] += LR * hidenToOutputError[i];
		}
	}

	void hidenToOutError(int targetIndex) {
		for (int i = 0; i < outputNeuronsAmount; i++) {
			hidenToOutputError[i] = (learningTargets[targetIndex] - outputLayer[i]) * directiveSigmoid(outputLayer[i]);
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
		double lrOutput = 0.01;
		for (int i = 0; i < hidenNeuronsAmount; i++) {
			for (int j = 0; j < outputNeuronsAmount; j++) {
				hidenToOutputWeights[i][j] += NewVelocity(lrOutput,hidenToOutputError[j], hidenLayer[i], hidenToOutputLayerVelocity[i][j]);

			}
		}
		double lrInput = 0.01;
		for (int i = 0; i < inputNeuronsAmount; i++) {
			for (int j = 0; j < hidenNeuronsAmount; j++) {
				InputToHiddenWeights[i][j] += NewVelocity(lrInput,inputToHidenError[j], inputLayer[i], inputToHiddenLayerVelocity[i][j]);
			}
		}

	}

	double MSE(int targetIndex) {
		double errorSum = 0;
		for (int i = 0; i < outputNeuronsAmount; i++) {
			if (learningTargets.size() <= 0) {
				throw Errors::DIVISIONBYZERO;
			}
			errorSum += std::pow((learningTargets[targetIndex] - outputLayer[i]),2);
		}

		return (1.0 / outputNeuronsAmount) * errorSum; 
	}

	double RMSE(int targetIndex) {
		return std::sqrt(MSE(targetIndex));
	}

	void ShowHidenWeights() {
		for (int i = 0; i < InputToHiddenWeights.size();i++) {
			for (int j = 0; j < InputToHiddenWeights[i].size();j ++) {
				std::cout << "InputToHiddenWeight[" << i << "][" << j << "]\t" << InputToHiddenWeights[i][j] << std::endl;
			}
		}
		std::cout << "**********************************************************************" << std::endl;
		for (int i = 0; i < hidenToOutputWeights.size(); i++) {
			for (int j = 0; j < hidenToOutputWeights[i].size(); j++) {
				std::cout << "HidenToOutputWeight[" << i << "][" << j << "]" << hidenToOutputWeights[i][j]<< std::endl;
			}
		}
		char ch1 = _getch();
	}

	void Learning() {
		
		std::vector<double> res(1, 0);
		int i = 0;
		epoch = 0;
		while(1){

			double epochError = 0.0;
			for (int i = 0; i < learningLogs.size(); i++) {
				SetInputLayer(vectorize->TFxIDF(learningLogs[i], inputNeuronsAmount));
				/*std::cout << "TF-IDF: ";
				for (auto n : vectorize->TFxIDF(learningLogs[i], inputNeuronsAmount)) {
					std::cout << n;
				}*/
				//std::cout << std::endl;
				//char ch = _getch();
				std::cout << "\nEpoch: " << epoch << std::endl;
				std::cout << "Learning on log:\t" << learningLogs[i] << "\tDanger:\t" << learningTargets[i] << std::endl;
				InputToHiddenLayerProccess();
				HiddenToOutputLayerProccess();
				res = GetOutputLayer();
				for (auto n : res) {
					std::cout << "Result: " << n << "\tError: " << RMSE(i) << std::endl;
				}
				epochError += RMSE(i);
				hidenToOutError(i);
				inToHidenError();
				UpdateBias();
				UpdateWeights();
				CleanInputLayer();
			}
			if ((epochError / learningLogs.size() * 100) < 5) {
				std::cout << "Learning done on epoch " << epoch << std::endl;
				break;
			}
			epoch++;
			Sleep(1000);
			//ShowHidenWeights();
			std::cout << "\nNew epoch " << epoch << std::endl;
		}
	}
	~Perceptron() {
		delete vectorize;
		delete dataset;
	}
};

int main() {
	setlocale(LC_ALL, "ru");
	Perceptron p(100, 80, 1);
	Tests test;
	std::vector<std::string> log = test.GetTestLog();
	
	try {
		for (int i = 0; i < test.GetTestSetSize(); i++) {
			std::cout << "Log: " << log[i] << std::endl;
			p.ProccedString(log[i]);
			p.InputToHiddenLayerProccess();
			p.HiddenToOutputLayerProccess();
			std::vector<double> res = p.GetOutputLayer();
			std::cout << "Result: " << res[0] << std::endl;
			test.Testing(p.GetOutputLayer(), i);
			p.CleanInputLayer();
		}
		std::cout << "Proccent of corretc answers:\t" << test.GetSuccessProcent() << "%" << std::endl;
	}
	catch (const Errors &e) {
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