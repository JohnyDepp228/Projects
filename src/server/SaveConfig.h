
#ifndef SAVECONFIG_H
#define SAVECONFIG_H

#include <vector>
#include <string>
#include <fstream>
#include "Errors.h"

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
		const unsigned int& inputNeuronsAmount, const unsigned int& hidenNeuronsAmount, const unsigned int& outputNeuronsAmount);
	SavePerceprtonConfig();

	void SaveToFile(std::string path, const std::vector<std::vector<double>>& InputToHiddenWeights, const std::vector<std::vector<double>>& hidenToOutputWeights,
		const std::vector<std::vector<double>>& hidenToOutputLayerVelocity, const std::vector<std::vector<double>>& inputToHiddenLayerVelocity,
		const std::vector<double>& inputToHiddenLayerBias, const std::vector<double>& hidenToOutputLayerBias,
		const unsigned int& inputNeuronsAmount, const unsigned int& hidenNeuronsAmount, const unsigned int& outputNeuronsAmount);

	bool GetFromFile(std::string path);

	void SetConfig(std::vector<std::vector<double>>& InputToHiddenWeights, std::vector<std::vector<double>>& hidenToOutputWeights,
		std::vector<std::vector<double>>& hidenToOutputLayerVelocity, std::vector<std::vector<double>>& inputToHiddenLayerVelocity,
		std::vector<double>& inputToHiddenLayerBias, std::vector<double>& hidenToOutputLayerBias,
		unsigned int& inputNeuronsAmount, unsigned int& hidenNeuronsAmount, unsigned int& outputNeuronsAmount);

	void SetConfig(const SavePerceprtonConfig& config);
};


#endif