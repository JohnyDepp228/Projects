#include "perceptron.h"


Perceptron::Perceptron(const unsigned int& inputNeuronsAmount, const unsigned int& hidenNeuronsAmount, const unsigned int& outputNeuronsAmount) {
	this->inputNeuronsAmount = inputNeuronsAmount;
	this->hidenNeuronsAmount = hidenNeuronsAmount;
	this->outputNeuronsAmount = outputNeuronsAmount;
}

void Perceptron::Start() {
	SavePerceprtonConfig config;
	epoch = 0;
	dataset = new Dataset(2000);
	this->learningTargets = dataset->GetLogsDanger();
	this->learningLogs = dataset->GetLogs();
	path = "weights.txt";
	vectorize = new WordsVectorize(learningLogs, dataset->methods, dataset->pages, dataset->protocol,
		dataset->agent, dataset->SQL, dataset->XSS, dataset->Win);
	if (config.GetFromFile(path) && vectorize->FindFileIDF()) {
		config.SetConfig
		(
			this->InputToHiddenWeights, this->hidenToOutputWeights,
			this->hidenToOutputLayerVelocity, this->inputToHiddenLayerVelocity,
			this->inputToHiddenLayerBias, this->hidenToOutputLayerBias,
			this->inputNeuronsAmount, this->hidenNeuronsAmount, this->outputNeuronsAmount
		);
		this->inputLayer = std::vector<double>(this->inputNeuronsAmount, 0);
		this->hidenLayer = std::vector<double>(this->hidenNeuronsAmount, 0);
		this->hidenLayerBeforeReLu = std::vector<double>(this->hidenNeuronsAmount, 0);
		this->outputLayer = std::vector<double>(this->outputNeuronsAmount, 0);
		vectorize->ReadIDFFromFile();
	}
	else {
		std::cout << "IDF find\t" << vectorize->FindFileIDF() << "\tWeight find\t" << config.GetFromFile(path) << std::endl;
		this->inputLayer = std::vector<double>(this->inputNeuronsAmount, 0);
		this->hidenLayer = std::vector<double>(this->hidenNeuronsAmount, 0);
		this->hidenLayerBeforeReLu = std::vector<double>(this->hidenNeuronsAmount, 0);
		this->outputLayer = std::vector<double>(this->outputNeuronsAmount, 0);

		this->inputToHiddenLayerBias = std::vector<double>(this->hidenNeuronsAmount, 0.01);
		this->hidenToOutputLayerBias = std::vector<double>(this->outputNeuronsAmount, 0);

		this->inputToHidenError = std::vector<double>(this->hidenNeuronsAmount, 0.0);
		this->hidenToOutputError = std::vector<double>(this->outputNeuronsAmount, 0.0);

		this->hidenToOutputLayerVelocity = std::vector<std::vector<double>>(this->hidenNeuronsAmount, std::vector<double>(this->outputNeuronsAmount, 0));
		this->inputToHiddenLayerVelocity = std::vector<std::vector<double>>(this->inputNeuronsAmount, std::vector<double>(this->hidenNeuronsAmount, 0));

		this->InputToHiddenWeights = std::vector<std::vector<double>>(this->inputNeuronsAmount, std::vector<double>(this->hidenNeuronsAmount, 0));
		this->hidenToOutputWeights = std::vector<std::vector<double>>(this->hidenNeuronsAmount, std::vector<double>(this->outputNeuronsAmount, 0));

		Perceptron::InitWeights(this->InputToHiddenWeights);
		Perceptron::InitWeights(this->hidenToOutputWeights);
		vectorize->SetIDF();

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

	for (unsigned int j = 0; j < hidenNeuronsAmount; j++) {
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
		outputLayer[j] = Sigmoid(outputLayer[j] + hidenToOutputLayerBias[j]);

	}
}

void Perceptron::CleanInputLayer() {
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

int Perceptron::hash(std::string key) {
	std::hash<std::string> hasher;
	return hasher(key) % inputNeuronsAmount;
}

void Perceptron::ProccedString(std::string str) {
	/*for (auto n : vectorize->TFxIDF(str, inputNeuronsAmount)) {
		std::cout << "TF-IDF:" << n << std::endl;
	}*/
	SetInputLayer(vectorize->TFxIDF(str, inputNeuronsAmount));
}

double Perceptron::NewVelocity(double LR, const double& neuronError, const double& neuronInput, double& oldVelocity) {
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

void Perceptron::hidenToOutError(int targetIndex) {
	for (int i = 0; i < outputNeuronsAmount; i++) {
		hidenToOutputError[i] = (learningTargets[targetIndex] - outputLayer[i]) * directiveSigmoid(outputLayer[i]);
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
			hidenToOutputWeights[i][j] += NewVelocity(lrOutput, hidenToOutputError[j], hidenLayer[i], hidenToOutputLayerVelocity[i][j]);

		}
	}
	double lrInput = 0.01;
	for (int i = 0; i < inputNeuronsAmount; i++) {
		for (int j = 0; j < hidenNeuronsAmount; j++) {
			InputToHiddenWeights[i][j] += NewVelocity(lrInput, inputToHidenError[j], inputLayer[i], inputToHiddenLayerVelocity[i][j]);
		}
	}

}

double Perceptron::MSE(int targetIndex) {
	double errorSum = 0;
	for (int i = 0; i < outputNeuronsAmount; i++) {
		if (learningTargets.size() <= 0) {
			throw Errors::DIVISIONBYZERO;
		}
		errorSum += std::pow((learningTargets[targetIndex] - outputLayer[i]), 2);
	}

	return (1.0 / outputNeuronsAmount) * errorSum;
}

double Perceptron::RMSE(int targetIndex) {
	return std::sqrt(MSE(targetIndex));
}

void Perceptron::ShowHidenWeights() {
	for (int i = 0; i < InputToHiddenWeights.size(); i++) {
		for (int j = 0; j < InputToHiddenWeights[i].size(); j++) {
			std::cout << "InputToHiddenWeight[" << i << "][" << j << "]\t" << InputToHiddenWeights[i][j] << std::endl;
		}
	}
	std::cout << "**********************************************************************" << std::endl;
	for (int i = 0; i < hidenToOutputWeights.size(); i++) {
		for (int j = 0; j < hidenToOutputWeights[i].size(); j++) {
			std::cout << "HidenToOutputWeight[" << i << "][" << j << "]" << hidenToOutputWeights[i][j] << std::endl;
		}
	}
	char ch1 = _getch();
}

void Perceptron::Learning() {

	std::vector<double> res(1, 0);
	int i = 0;
	epoch = 0;
	while (1) {

		double epochError = 0.0;
		for (int i = 0; i < learningLogs.size(); i++) {
			std::cout << "\nEpoch: " << epoch << std::endl;
			std::cout << "Learning on log:\t" << learningLogs[i] << "\tDanger:\t" << learningTargets[i] << std::endl;
			ProccedString(learningLogs[i]);
			InputToHiddenLayerProccess();
			HiddenToOutputLayerProccess();
			res = GetOutputLayer();
			for (auto n : res) {
				std::cout << "Result: " << n << "\tError: " << RMSE(i) << std::endl;
			}
			epochError += RMSE(i);
			hidenToOutError(i);
			inToHidenError();
			UpdateBias();
			UpdateWeights();
			/*for (auto n : inputLayer) {
				std::cout << "\nInput layer\t" << n;
			}*/
			CleanInputLayer();
		}
		if ((epochError / learningLogs.size() * 100) < 5) {
			std::cout << "Learning done on epoch " << epoch << std::endl;
			break;
		}
		epoch++;
		Sleep(1000);
		std::cout << "\nNew epoch " << epoch << std::endl;
		//ShowHidenWeights();
	}
}

bool Perceptron::FullProcess(std::string str) {
	ProccedString(str);
	InputToHiddenLayerProccess();
	HiddenToOutputLayerProccess();
	std::vector<double> res = GetOutputLayer();
	std::cout << "Result: " << res[0] << std::endl;
	return (res[0] * 100) > 50;
}