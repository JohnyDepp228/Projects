#include <iostream>
#include <vector>
#include <random>
#include <string>



class Perceptron {
private:

	std::vector<std::vector<double>> InputToHiddenWeights;
	std::vector<std::vector<double>> hidenToOutputWeights;
	std::vector<std::vector<double>> hidenToOutputLayerVelocity;
	std::vector<std::vector<double>> inputToHiddenLayerVelocity;

	std::vector<double> inputLayer;
	std::vector<double> hidenLayer;
	std::vector<double> outputLayer;
	std::vector<double> inputToHiddenLayerBias;
	std::vector<double> hidenToOutputLayerBias;
	std::vector<double> inputToHidenError;
	std::vector<double> hidenToOutputError;
	std::vector<double> learningTargets;

	double LR = 0.01;
	double inertia = 0.1;

	unsigned int inputNeuronsAmount;
	unsigned int hidenNeuronsAmount;
	unsigned int outputNeuronsAmount;



public:
	Perceptron(const unsigned int& inputNeuronsAmount, const unsigned int& hidenNeuronsAmount, const unsigned int& outputNeuronsAmount,
		std::vector<double> targets) {
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

	void hidenToOutError(double target) {
		for (int i = 0; i < outputNeuronsAmount; i++) {
			hidenToOutputError[i] = (outputLayer[i] - target) * directiveSigmoid(outputLayer[i]);
		}
	}

	void inToHidenError(double target) {
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
	}
};

int main() {
	std::vector<double> res2(1, 0);
	Perceptron p(10, 7, 1, res2);
	std::string test = { "Hello world peace and apple god dog layer cat weight" };
	std::vector<double> res(1, 0);
	p.ProccedString(test);
	p.InputToHiddenLayerProccess();
	p.HiddenToOutputLayerProccess();
	res = p.GetOutputLayer();

	for (auto n : res) {
		std::cout << "Result: " << n;
	}

	return 0;
}