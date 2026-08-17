#ifndef PERCEPTRON_H
#define PERCEPTRON


#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <Windows.h>
#include "DatasetGenerator.h"
#include "SentenceVectorize.h"
#include "SaveConfig.h"
#include "Errors.h"

class Perceptron {
private:

	std::vector<std::vector<double>> InputToHiddenWeights;
	std::vector<std::vector<double>> hidenToOutputWeights;
	std::vector<std::vector<double>> hidenToOutputLayerVelocity;
	std::vector<std::vector<double>> inputToHiddenLayerVelocity;

	std::vector<double> inputLayer;
	std::vector<double> hidenLayer;
	std::vector<double> hidenLayerBeforeReLu;
	std::vector<double> outputLayer;

	std::vector<double> inputToHiddenLayerBias;
	std::vector<double> hidenToOutputLayerBias;

	std::vector<double> inputToHidenError;
	std::vector<double> hidenToOutputError;


	std::vector<double> learningTargets;
	std::vector<std::string> learningLogs;

	double LR = 0.5;
	double inertia = 0.01;

	unsigned int inputNeuronsAmount;
	unsigned int hidenNeuronsAmount;
	unsigned int outputNeuronsAmount;
	unsigned int epoch;

	std::string weightsPath;
	std::string idfPath;

	WordsVectorize* vectorize;

	Dataset* dataset;

public:
	Perceptron(const unsigned int& inputNeuronsAmount, const unsigned int& hidenNeuronsAmount, const unsigned int& outputNeuronsAmount,std::string weightsPath,std::string idfPath);

	double LeakyReLu(const double& data) {
		return data > 0 ? data : data * 0.01;
	}

	double directiveLeakyReLu(const double& data) {
		return data > 0 ? 1.0 : 0.01;
	}

	double Sigmoid(const double& data) {
		return 1.0 / (1.0 + std::exp(-data));
	}

	double directiveSigmoid(const double& data) {
		return data * (1.0 - data);
	}

	void SetInputLayer(const std::vector<double>& input);

	void InputToHiddenLayerProccess();

	void HiddenToOutputLayerProccess();

	void CleanInputLayer();

	void InitWeights(std::vector<std::vector<double>>& layer);

	double Weight(double leftBoard, double rightBoard);

	void ProccedString(std::string str);

	double NewVelocity(double LR, const double& neuronError, const double& neuronInput, double& oldVelocity);

	void UpdateBias();

	void hidenToOutError(int targetIndex);

	void inToHidenError();

	double SumVec(std::vector<double> vec);

	void UpdateWeights();

	double MSE(int targetIndex);

	double RMSE(int targetIndex);

	void ShowHidenWeights();

	void Learning();

	bool FullProcess(std::string);

	void Start();

	std::vector<double> GetOutputLayer() const {
		return outputLayer;
	}

	~Perceptron() {
		delete vectorize;
		delete dataset;
	}

};



#endif