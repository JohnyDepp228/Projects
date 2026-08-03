#include "SaveConfig.h"


SavePerceprtonConfig::SavePerceprtonConfig(std::vector<std::vector<double>> InputToHiddenWeights, std::vector<std::vector<double>> hidenToOutputWeights,
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
SavePerceprtonConfig::SavePerceprtonConfig() {
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

void SavePerceprtonConfig::SaveToFile(std::string path, const std::vector<std::vector<double>>& InputToHiddenWeights, const std::vector<std::vector<double>>& hidenToOutputWeights,
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

bool SavePerceprtonConfig::GetFromFile(std::string path) {
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

void SavePerceprtonConfig::SetConfig(std::vector<std::vector<double>>& InputToHiddenWeights, std::vector<std::vector<double>>& hidenToOutputWeights,
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

void SavePerceprtonConfig::SetConfig(const SavePerceprtonConfig& config)
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