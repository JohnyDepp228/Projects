#include "perceptron.h"


Perceptron::Perceptron(const unsigned int &inputNeuronsAmount,const unsigned int &hidenNeuronsAmount,const unsigned int &outputNeuronsAmount) {
		SavePerceprtonConfig config;
		epoch = 0;
		unsigned int uniqueWordsInDataset = 37;
		path = ".";
		if (config.GetFromFile(path)) {
				config.SetConfig
				(
					this->InputToHiddenWeights, this->hidenToOutputWeights,
					this->hidenToOutputLayerVelocity, this->inputToHiddenLayerVelocity, 
					this->inputToHiddenLayerBias, this->hidenToOutputLayerBias,
					this->inputNeuronsAmount, this->hidenNeuronsAmount, this->outputNeuronsAmount
				);
				this->inputLayer = std::vector<double>(this->inputNeuronsAmount, 0.0);
				this->hidenLayer = std::vector<double>(this->hidenNeuronsAmount, 0.0);
				this->hidenLayerBeforeReLu = std::vector<double>(this->hidenNeuronsAmount, 0.0);
				this->outputLayer = std::vector<double>(this->outputNeuronsAmount, 0.0);
		}
		else {
			this->inputNeuronsAmount = inputNeuronsAmount;
			this->hidenNeuronsAmount = hidenNeuronsAmount;
			this->outputNeuronsAmount = outputNeuronsAmount;

			this->inputLayer.resize(this->inputNeuronsAmount, 0.0);
			this->hidenLayer.resize(this->hidenNeuronsAmount, 0.0);
			this->hidenLayerBeforeReLu.resize(this->hidenNeuronsAmount, 0.0);
			this->outputLayer.resize(this->outputNeuronsAmount, 0.0);
			

			inputToHiddenLayerBias = std::vector<double>(hidenNeuronsAmount, 0.01);
			hidenToOutputLayerBias = std::vector<double>(outputNeuronsAmount, 0.0);

			inputToHidenError = std::vector<double>(hidenNeuronsAmount, 0.0);
			hidenToOutputError = std::vector<double>(outputNeuronsAmount, 0.0);

			hidenToOutputLayerVelocity = std::vector<std::vector<double>>(hidenNeuronsAmount, std::vector<double>(outputNeuronsAmount, 0.0));
			inputToHiddenLayerVelocity = std::vector<std::vector<double>>(inputNeuronsAmount, std::vector<double>(hidenNeuronsAmount, 0.0));

			InputToHiddenWeights = std::vector<std::vector<double>>(inputNeuronsAmount, std::vector<double>(hidenNeuronsAmount, 0.0));
			hidenToOutputWeights = std::vector<std::vector<double>>(hidenNeuronsAmount, std::vector<double>(outputNeuronsAmount, 0.0));

			Perceptron::InitWeights(InputToHiddenWeights);
			Perceptron::InitWeights(hidenToOutputWeights);
			
			Perceptron::Learning();
			config.SaveToFile(path, this->InputToHiddenWeights, this->hidenToOutputWeights,
				this->hidenToOutputLayerVelocity, this->inputToHiddenLayerVelocity,
				this->inputToHiddenLayerBias, this->hidenToOutputLayerBias,
				this->inputNeuronsAmount, this->hidenNeuronsAmount, this->outputNeuronsAmount);
		}
	}

	void Perceptron::SetInputLayer(const std::vector<double>& input) {
		std::fill(inputLayer.begin(), inputLayer.end(), 0.0);
		if (input.size() <= inputLayer.size()) {
			inputLayer = input;
		}
		else {
			inputLayer.assign(input.begin(), input.begin() + inputLayer.size());
		}
	}

	void Perceptron::InputToHiddenLayerProccess() {
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

	void Perceptron::HiddenToOutputLayerProccess() {
		std::fill(outputLayer.begin(), outputLayer.end(), 0.0);
		for (int i = 0; i < hidenNeuronsAmount; i++) {
			for (int j = 0; j < outputNeuronsAmount; j++) {
				outputLayer[j] += (hidenLayer[i] * hidenToOutputWeights[i][j]);
			}
		}
		for (int j = 0; j < outputNeuronsAmount; j++) {
			outputLayer[j] = (outputLayer[j] + hidenToOutputLayerBias[j]);
		}

		for (int j = 0; j < outputNeuronsAmount; j++) {
			outputLayer[j] = SoftMax(outputLayer[j]);
		}
	}

	void Perceptron:: CleanInputLayer() {
		std::fill(inputLayer.begin(), inputLayer.end(), 0.0);
	}

	void Perceptron::InitWeights(std::vector<std::vector<double>>& layer) {
		for (auto& colls : layer) {
			for (auto& rows : colls) {
				rows = Weight(-0.5, 0.5);
			}
		}
	}


	double Perceptron::Weight(double leftBoard, double rightBoard) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(leftBoard, rightBoard);
		return dis(gen);
	}



	double Perceptron::NewVelocity(double LR,const double& neuronError, const double& neuronInput,double &oldVelocity) {
		return oldVelocity = (inertia * oldVelocity) + (LR * neuronError * neuronInput);
	}

	void Perceptron::UpdateBias() {
		for (int i = 0; i < inputToHiddenLayerBias.size(); i++) {
			inputToHiddenLayerBias[i] += LR * inputToHidenError[i];
		}

		for (int i = 0; i < hidenToOutputLayerBias.size(); i++) {
			hidenToOutputLayerBias[i] += LR * hidenToOutputError[i];
		}
	}

	void Perceptron::hidenToOutError(double target) {
		std::vector<double> targetMap(outputNeuronsAmount, 0.0);
		targetMap[target] = 1;
		for (int i = 0; i < outputNeuronsAmount; i++) {
			hidenToOutputError[i] = (targetMap[i] - outputLayer[i]);
		}
	}

	void Perceptron::inToHidenError() {
		for (int i = 0; i < hidenNeuronsAmount; i++) {
			double errorSum = 0;
			for (int j = 0; j < outputNeuronsAmount; j++) {
				errorSum += hidenToOutputError[j] * hidenToOutputWeights[i][j];
			}
			inputToHidenError[i] = errorSum * directiveReLu(hidenLayerBeforeReLu[i]);
		}
	}

	double Perceptron::SumVec(std::vector<double> vec) {
		double sum = 0;
		for (auto i : vec) {
			sum += i;
		}

		return sum;
	}

	void Perceptron::UpdateWeights() {
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

	double Perceptron::CrossEntrypy(double target) {
		return -(std::log(outputLayer[target]));
	}

	void Perceptron::Learning() {
		std::vector<double> res;
		int datasetSize = 1000;
		GenerateDataset dataset(datasetSize);
		int i = 0;
		epoch = 0;
		while(1){
			double epochError = 0.0;
			for (int i = 0; i < datasetSize; i++) {
				SetInputLayer(dataset.GetMap(i));
				InputToHiddenLayerProccess();
				HiddenToOutputLayerProccess();
				res = GetOutputLayer();
				epochError += CrossEntrypy(dataset.GetAnswer(i));
				hidenToOutError(dataset.GetAnswer(i));
				inToHidenError();
				UpdateBias();
				UpdateWeights();
				CleanInputLayer();
			}
			if ((epochError / datasetSize * 100) < 5) {
				std::cout << "Learning done on epoch " << epoch << std::endl;
				break;
			}
			epoch++;
			std::cout << "Epoch\t" << epoch << "\tLoss:\t" << (epochError / datasetSize * 100) << std::endl;
			Sleep(1000);
		}
	}

	bool Perceptron::FullProcess(const std::vector<double> &input) {
		SetInputLayer(input);
		InputToHiddenLayerProccess();
		HiddenToOutputLayerProccess();
		std::vector<double> res = GetOutputLayer();
		return (res[0] * 100) > 50;
	}

	int Perceptron::GetAnswer() {
		auto it = *std::max_element(outputLayer.begin(), outputLayer.end());
		int res = 0;
		for (int i = 0; i < outputLayer.size(); i++) {

			if (outputLayer[i] == it) {
				res = i;
				break;
			}
		}
		return res;
	}