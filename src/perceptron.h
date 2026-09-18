#ifndef PERCEPTRON_H
#define PERCEPTRON


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

	std::string path;

	Dataset* dataset;

public:
	Perceptron(const unsigned int& inputNeuronsAmount, const unsigned int& hidenNeuronsAmount, const unsigned int& outputNeuronsAmount);

	double ReLu(const double& data) {
		//return data > 0 ? data : 0;
		return data > 0 ? data : data * 0.01; //leakyReLu
	}

	double directiveReLu(const double& data) {
		//return data > 0 ? 1.0 : 0;
		return data > 0 ? 1.0 : 0.01; //Leaky ReLu
	}

	double Sigmoid(const double& data) {
		return 1.0 / (1.0 + std::exp(-data));
	}

	double directiveSigmoid(const double& data) {
		return data * (1.0 - data); // data after sigmoid
	}

	double OutLayerSum() {
		double res = 0.0;
		for (const auto& n : outputLayer) {
			res += std::exp(n);
		}
		return res;
	}

	double SoftMax(const double& data) {
		double sum = OutLayerSum();
		return std::exp(data) / sum;
	}

	void SetInputLayer(const std::vector<double>& input);

	void InputToHiddenLayerProccess();

	void HiddenToOutputLayerProccess();

	void CleanInputLayer();

	void InitWeights(std::vector<std::vector<double>>& layer);

	double Weight(double leftBoard, double rightBoard);

	double NewVelocity(double LR, const double& neuronError, const double& neuronInput, double& oldVelocity);

	void UpdateBias();

	void hidenToOutError(double target);

	void inToHidenError();

	double SumVec(std::vector<double> vec);

	void UpdateWeights();

	double CrossEntrypy(double target);

	void Learning();

	bool FullProcess(const std::vector<double>& input);

	std::vector<double> GetOutputLayer() const {
		return outputLayer;
	}

	int GetAnswer();

	~Perceptron() {
		delete dataset;
	}

};



#endif

