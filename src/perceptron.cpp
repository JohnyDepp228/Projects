#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <string>

using std::vector;


class Perceptron {
private:
	vector<vector<double>> outputWeights;
	vector<vector<double>> hidenWeights;
	vector<vector<double>> hVelocity;
	vector<vector<double>> outVelocity;
	vector<double> inputLayer;
	vector<double> hiddenLayer;
	vector<double> outputLayer;
	unsigned int inputNeurons;
	unsigned int hidenNeurons;
	unsigned int outputNeurons;
	double bias = 1.0;
	double LR = 0.001;
	double inertia = 0.8;


public:
	Perceptron(unsigned int inputNeurons, unsigned int hidenNeurons, unsigned int outputNeurons, std::string input) {
		this->inputNeurons = inputNeurons;
		this->hidenNeurons = hidenNeurons;
		this->outputNeurons = outputNeurons;
		inputLayer = vector<double>(inputNeurons, 0.0);
		hiddenLayer = vector<double>(hidenNeurons, 0.0);
		outputLayer = vector<double>(outputNeurons, 0.0);
		outputWeights = vector<vector<double>>(inputNeurons, vector<double>(hidenNeurons, 0.0));
		hVelocity = vector<vector<double>>(inputNeurons, vector<double>(hidenNeurons, 0.0));
		hidenWeights = vector<vector<double>>(hidenNeurons, vector<double>(outputNeurons, 0.0));
		outVelocity = vector<vector<double>>(hidenNeurons, vector<double>(outputNeurons, 0.0));
		InitWeights(outputWeights);
		InitWeights(hidenWeights);
	}

	void InitWeights(vector<vector<double>>& weights) {
		for (auto& rows : weights) {
			for (auto& colls : rows) {
				colls = Weight(0.0, 1.0);
			}
		}
	}

	double Weight(int leftBoard, int rightBoard) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(leftBoard, rightBoard);
		return dis(gen);
	}

	void HiddenLayer() {
		for (int neuronIndex = 0; neuronIndex < hiddenLayer.size(); neuronIndex++) {
			for (int i = 0; i < inputLayer.size(); i++) {
				hiddenLayer[neuronIndex] += inputLayer[i] * outputWeights[i][neuronIndex];
			}
		}
	}

	void OutputLayer() {
		for (int neuronIndex = 0; neuronIndex < outputLayer.size(); neuronIndex++) {
			for (int i = 0; i < inputLayer.size(); i++) {
				outputLayer[neuronIndex] += hiddenLayer[i] * hidenWeights[i][neuronIndex];
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

	vector<double> GetOutput() const {
		return outputLayer;
	}

	void CleaarInput() {
		inputLayer = vector<double>(inputNeurons, 0.0);
	}

	double Relu(double data) {
		return (data > 0.0) ? data : 0.0;
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

	vector<double> ProccedString(std::string str) {
		vector<std::string> words(inputNeurons, "0");
		unsigned int i = 0;
		unsigned int j = 0;
		while (str[i] != '\0') {
			if (str[i] == ' ') {
				j++;
				i++;
			}
			words[j] += str[i];
			i++;
		}

		for (int i = 0; i < words.size(); i++) {
			inputLayer[hash(words[i])]++;
		}
	}


	void TrainInputHiden() {
		for (int i = 0; i < hidenWeights.size(); i++) {
			for (int j = 0; j < hidenWeights[0].size(); j++) {
				hidenWeights[i][j] = hidenWeights[i][j] - hVelocity[i][j];
			}
		}
	}

	void TrainHidenOutput() {
		for (int i = 0; i < outputWeights.size(); i++) {
			for (int j = 0; j < outputWeights[0].size(); j++) {
				outputWeights[i][j] = outputWeights[i][j] - outVelocity[i][j];
			}
		}
	}

	double hiddenVelocity() {
		for (int i = 0; i < inputNeurons; i++) {
			for (int j = 0; j < hidenWeights[0].size(); i++) {

			}
		}
	}
};






int main()
{

}