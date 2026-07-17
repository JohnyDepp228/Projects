#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <fstream>

enum Errors {
	FILEREAD = 1,
	FILEWRITE
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


			};
		if (!fout.write((char*)&config, sizeof(SavePerceprtonConfig))) {
			throw Errors::FILEWRITE;
		}
		fout.close();
	}

	bool GetFromFile(std::string path) {
		SavePerceprtonConfig config;
		std::ifstream fin;
		fin.open(path, std::ios::binary);
		if (fin.is_open()) {
			if (!fin.read((char*)&config, sizeof(SavePerceprtonConfig))) {
				throw Errors::FILEREAD;
			};
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
		InputToHiddenWeights = this->hidenToOutputWeights;
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
		this->InputToHiddenWeights = config.hidenToOutputWeights;
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
	std::vector<double> outputLayer;

	std::vector<double> inputToHiddenLayerBias;//save
	std::vector<double> hidenToOutputLayerBias;//save

	std::vector<double> inputToHidenError;
	std::vector<double> hidenToOutputError;


	std::vector<double> learningTargets;

	double LR = 0.01;
	double inertia = 0.1;

	unsigned int inputNeuronsAmount;
	unsigned int hidenNeuronsAmount;
	unsigned int outputNeuronsAmount;
	unsigned int epoch;



public:
	Perceptron(const unsigned int& inputNeuronsAmount, const unsigned int& hidenNeuronsAmount, const unsigned int& outputNeuronsAmount,
		std::vector<double> targets) {
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
			inputLayer = std::vector<double>(this->inputNeuronsAmount, 0);
			hidenLayer = std::vector<double>(this->hidenNeuronsAmount, 0);
			outputLayer = std::vector<double>(this->outputNeuronsAmount, 0);
		}
		else {
			this->inputNeuronsAmount = inputNeuronsAmount;
			this->hidenNeuronsAmount = hidenNeuronsAmount;
			this->outputNeuronsAmount = outputNeuronsAmount;
			this->learningTargets = targets;

			inputLayer = std::vector<double>(inputNeuronsAmount, 0);
			hidenLayer = std::vector<double>(hidenNeuronsAmount, 0);
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
		int sum = 0;
		for (char c : key) {
			sum += (int)c;
		}
		int lastTwoDigits = sum % 100;
		long long square = (long long)lastTwoDigits * lastTwoDigits;
		std::string squareStr = std::to_string(square);
		int mid;
		if (squareStr.length() <= 4) {
			mid = lastTwoDigits;
		}
		else {
			int start = (squareStr.length() - 4) / 2;
			std::string midStr = squareStr.substr(start, 4);
			mid = stoi(midStr);
		}
		return mid % inputNeuronsAmount;
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
			inputLayer[hash(word)]++;
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
			hidenToOutputError[i] = (outputLayer[i] - learningTargets[epoch]) * directiveSigmoid(outputLayer[i]);
		}
	}

	void inToHidenError() {
		for (int i = 0; i < hidenNeuronsAmount; i++) {
			double errorSum = 0;
			for (int j = 0; j < outputNeuronsAmount; j++) {
				errorSum += hidenToOutputError[j] * hidenToOutputWeights[i][j];
			}
			inputToHidenError[i] = errorSum * directiveReLu(hidenLayer[i]);
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

		epoch++;
	}

	double MSE() {
		double errorSum = 0;
		for (int i = 0; i < outputNeuronsAmount; i++) {
			errorSum += std::pow((outputLayer[i] - learningTargets[epoch]), 2);
		}

		return (1.0 / outputNeuronsAmount) * errorSum;
	}

	double RMSE() {
		return std::sqrt(MSE());
	}

	void Learning() {
		std::string test = { "Hello world peace and apple god dog layer cat weight" };
		std::vector<double> res(1, 0);
		int i = 0;
		while (i < learningTargets.size() - 1) {
			ProccedString(test);
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
	std::vector<double> res2(10000000, 1);
	std::string test = { "Hello world peace and apple god dog layer cat weight" };
	std::vector<double> res(1, 0);
	Perceptron p(100, 64, 1, res2);
	try {
		p.ProccedString(test);
		p.InputToHiddenLayerProccess();
		p.HiddenToOutputLayerProccess();
		res = p.GetOutputLayer();
		for (auto n : res) {
			std::cout << "Result: " << n << "\tError: " << p.RMSE() << std::endl;
		}
	}
	catch (const Errors& e) {
		if (e == Errors::FILEREAD) {
			std::cout << "Can't read perceptron weights" << std::endl;
		}
		else if (e == Errors::FILEWRITE) {
			std::cout << "Can't save perceptron weights" << std::endl;
		}
	}
	catch (...) {
		std::cout << "Unknown problem" << std::endl;
	}
	return 0;
}