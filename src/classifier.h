#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#pragma once

#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <numbers> 



class Classifier
{
public:
    Classifier();
    ~Classifier();

    std::vector<std::vector<double>> TransponMatrix(const std::vector<std::vector<double>>& matrix);

	void InputErrorsForBackprop();

	double Weight(const double& leftBoard, const double& rightBoard);

	void InitMatrixWeights(std::vector < std::vector < double>>& matrix);

	double VecSum(const std::vector<double>& vec);

	double OutLayerSum();

	std::vector<double> HiddenLayersCalcul(const std::vector<double>& layer, const std::vector<std::vector<double>>& layerWeights
		, std::vector<double>& layerBeforeReLu, const std::vector<double>& layerBias);

	void Classification();

	void ApplySoftMax();

	double Linaer(double res);

	int FindCorrectOutNeuro();

	void ShowLayer(int i);

    void InitMatrix(std::vector < std::vector < double>>& matrix);

	//Learning
	void OutputLayerErrorCalcu(int correctNeuronIndex);

	void HiddenLayerErrorCalcu(const std::vector<double>& nextLayErrors, const std::vector<std::vector<double>>& nextlayerWeights,
		const std::vector<double>& beforeRelu, std::vector<double>& currentLayErrors);

	void Velocity(const std::vector<double>& currentLay, const std::vector<double> nextLayError, std::vector<std::vector<double>>& velocity);

	void UpdateLayerWieghts(const std::vector<std::vector<double>>& velocity, std::vector<std::vector<double>>& layerWeights);

	void UpdateBias(std::vector<double>& layerBias, const std::vector<double>& layerErrors);

	void ShowLayerOutWieght(const std::vector<std::vector<double>>& oldWeights);

	void LearningClassifier(int correctNeuronIndex);

	void ShowErrors(const std::vector<double>& error);

	//Activation Functions

	double LeakyReLu(double res);

	double DirectiveLeakyReLu(double res);

	double ReLu(double res);

	double DirectiveReLu(double res);

	double SoftMax(double res);

	double GeLu(double res);

	double DirectiveGeLu(double res);

	//Setters & Getters

	void SetFullyConnectedLayer(const std::vector<double>& vec);

	std::vector<double> GetOutLayer() const;

	std::vector<double> GetInputErrors() const;

private:
    int numOfFullyConnectedLayerNeurons = 512;
	int numOfFirstHiddenLayerNeurons = 460;
	int numOfSecondHiddenLayerNeurons = 230;
	int numOfThirdHiddenLayerNeurons = 130;
	int numOfOutputLayerNeurons = 100;
	double inertia = 0.4;
	double LR = 0.2;

	//layers
	std::vector<double> fullyConnectedLayer;
	std::vector<double> firstHiddenLayer;
	std::vector<double> secondHiddenLayer;
	std::vector<double> thirdHiddenLayer;
	std::vector<double> outputLayer;

	//Before ReLu
	std::vector<double> firstHiddenLayerBeforeReLu;
	std::vector<double> secondHiddenLayerBeforeReLu;
	std::vector<double> thirdHiddenLayerBeforeReLu;

	//Errors
	std::vector<double> outputLayerErrors;
	std::vector<double> fullyconnectedLayerErrors;
	std::vector<double> firstHiddenLayerErrors;
	std::vector<double> secondHiddenLayerErrors;
	std::vector<double> thirdHiddenLayerErrors;

	//Weights
	std::vector<std::vector<double>> fullConToFirstHiddenWeights;
	std::vector<std::vector<double>> firstToSecondHiddenWeights;
	std::vector<std::vector<double>> secondToThirdHiddenWeights;
	std::vector<std::vector<double>> thirdHiddenToOutputWeights;

	//Velocity
	std::vector<std::vector<double>> fullConToFirstHiddenVelocity;
	std::vector<std::vector<double>> firstToSecondHiddenVelocity;
	std::vector<std::vector<double>> secondToThirdHiddenVelocity;
	std::vector<std::vector<double>> thirdHiddenToOutputVelocity;

	//Bias
	std::vector<double> outputLayerBias;
	std::vector<double> firstHiddenLayerBias;
	std::vector<double> secondHiddenLayerBias;
	std::vector<double> thirdHiddenLayerBias;


	double(Classifier::* ActivationFunction)(double);

	double bias = 0.3;
};

#endif