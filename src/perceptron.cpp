#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <string>
#include <fstream>

using std::vector;

class Perceptron {
private:
	vector<vector<double>> inputToHiddenWeights;
	vector<vector<double>> hiddenToOutputWeights;
	vector<vector<double>> InputHidenVelocity;
	vector<vector<double>> HidenOutputVelocity;

	vector<double> inputLayer;
	vector<double> hiddenLayer;
	vector<double> outputLayer;
	vector<double> deltaOutput;

	unsigned int inputNeurons;
	unsigned int hidenNeurons;
	unsigned int outputNeurons;

	double bias = 1.0;
	double LR = 0.001;
	double inertia = 0.8;
	double main_error = 0;

public:
	Perceptron(unsigned int inputNeurons, unsigned int hidenNeurons, unsigned int outputNeurons) {
		this->inputNeurons = inputNeurons;
		this->hidenNeurons = hidenNeurons;
		this->outputNeurons = outputNeurons;

		inputLayer = vector<double>(inputNeurons, 0.0);
		hiddenLayer = vector<double>(hidenNeurons, 0.0);
		outputLayer = vector<double>(outputNeurons, 0.0);
		deltaOutput = vector<double>(outputNeurons, 0.0);

		inputToHiddenWeights = vector<vector<double>>(inputNeurons, vector<double>(hidenNeurons, 0.0));
		InputHidenVelocity = vector<vector<double>>(inputNeurons, vector<double>(hidenNeurons, 0.0));

		hiddenToOutputWeights = vector<vector<double>>(hidenNeurons, vector<double>(outputNeurons, 0.0));
		HidenOutputVelocity = vector<vector<double>>(hidenNeurons, vector<double>(outputNeurons, 0.0));

		InitWeights(inputToHiddenWeights);
		InitWeights(hiddenToOutputWeights);
	}

	void InitWeights(vector<vector<double>>& weights) {
		for (auto& rows : weights) {
			for (auto& colls : rows) {
				colls = Weight(-1.0, 1.0);
			}
		}
	}

	double Weight(double leftBoard, double rightBoard) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(leftBoard, rightBoard);
		return dis(gen);
	}

	void HiddenLayer() {
		std::fill(hiddenLayer.begin(), hiddenLayer.end(), 0.0);
		for (size_t neuronIndex = 0; neuronIndex < hiddenLayer.size(); neuronIndex++) {
			for (size_t i = 0; i < inputLayer.size(); i++) {
				hiddenLayer[neuronIndex] += inputLayer[i] * inputToHiddenWeights[i][neuronIndex];
			}

			hiddenLayer[neuronIndex] = Relu(hiddenLayer[neuronIndex]);
		}
	}

	void OutputLayer() {
		std::fill(outputLayer.begin(), outputLayer.end(), 0.0);
		for (size_t neuronIndex = 0; neuronIndex < outputLayer.size(); neuronIndex++) {
			for (size_t i = 0; i < hiddenLayer.size(); i++) {
				outputLayer[neuronIndex] += hiddenLayer[i] * hiddenToOutputWeights[i][neuronIndex];
			}
			outputLayer[neuronIndex] = Sigmoid(outputLayer[neuronIndex]);
		}
	}

	double Sigmoid(double data) {
		return 1.0 / (1.0 + std::exp(-data));
	}

	double deriativeSigmoid(double sigmoidRes) {
		return sigmoidRes * (1.0 - sigmoidRes);
	}

	double Relu(double data) {
		return (data > 0.0) ? data : 0.0;
	}

	double derivativeRelu(double data) {
		return (data > 0.0) ? 1.0 : 0.0;
	}

	vector<double> GetOutput() const {
		return outputLayer;
	}

	void CleaarInput() {
		std::fill(inputLayer.begin(), inputLayer.end(), 0.0);
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
		return mid % inputNeurons;
	}

	void ProccedString(std::string str) {
		if (str.empty()) return;
		vector<std::string> words;
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

	void TrainHidenOutput() {
		for (int i = 0; i < hidenNeurons; i++) {
			for (int j = 0; j < outputNeurons; j++) {
				FindHidenOutputVelocity(i, j);
				hiddenToOutputWeights[i][j] = hiddenToOutputWeights[i][j] - HidenOutputVelocity[i][j];
			}
		}
	}

	void TrainInputHiden() {
		for (int i = 0; i < inputNeurons; i++) {
			for (int j = 0; j < hidenNeurons; j++) {
				FindInputHidenVelocity(i, j);
				inputToHiddenWeights[i][j] = inputToHiddenWeights[i][j] - InputHidenVelocity[i][j];
			}
		}
	}

	void CalculateBackpropError(const vector<double>& targets) {
		for (int i = 0; i < outputNeurons; i++) {
			double error = outputLayer[i] - targets[i];
			deltaOutput[i] = error * deriativeSigmoid(outputLayer[i]);
		}
	}

	void Train(const vector<double>& targets) {

		HiddenLayer();
		OutputLayer();

		CalculateBackpropError(targets);

		main_error = outputLayer[0] - targets[0];
		if ((std::abs(main_error) * 100) > 15) {
			TrainHidenOutput();
			TrainInputHiden();
		}
	}

	double GetError() const {
		return main_error;
	}

	double Velocity(double oldVelocity, double ErrorToNeuron, double dataFromNeuron) {
		return (inertia * oldVelocity) + LR * ErrorToNeuron * dataFromNeuron;
	}

	void FindHidenOutputVelocity(int i, int j) {

		HidenOutputVelocity[i][j] = Velocity(HidenOutputVelocity[i][j], deltaOutput[j], hiddenLayer[i]);
	}

	void FindInputHidenVelocity(int i, int j) {

		double errorFromOutput = 0.0;
		for (int k = 0; k < outputNeurons; k++) {
			errorFromOutput += deltaOutput[k] * hiddenToOutputWeights[j][k];
		}

		double deltaHiddenJ = errorFromOutput * derivativeRelu(hiddenLayer[j]);
		InputHidenVelocity[i][j] = Velocity(InputHidenVelocity[i][j], deltaHiddenJ, inputLayer[i]);
	}

};

std::string GetStr(std::string filename) {
	std::ifstream inFile(filename);
	std::string str = "00";
	return str;
}





int main()
{
	Perceptron p(100, 64, 1);

	p.CleaarInput();


	std::string input = GetStr("C:\qerh");
	std::vector<double> targets;
	p.ProccedString(input);
	p.Train(targets);
	double error = p.GetError();

	int Epoch = 0;
	while ((error * 100) > 15) {
		input = GetStr("C:\qerh");
		p.ProccedString(input);
		p.Train(targets);
		error = p.GetError();
		Epoch++;
		std::cout << "Epoch №" << Epoch << "\tError: " << error << std::endl;
	}
	std::cout << "Learned" << std::endl;
	return 0;
}