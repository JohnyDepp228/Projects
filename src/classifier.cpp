#include "classifier.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ios>
#include <numeric>


Classifier::Classifier()
{
    //Layers
		outputLayer = std::vector<double>(numOfOutputLayerNeurons, 0.0);
		firstHiddenLayer = std::vector<double>(numOfFirstHiddenLayerNeurons, 0.0);
		secondHiddenLayer = std::vector<double>(numOfSecondHiddenLayerNeurons, 0.0);
		thirdHiddenLayer = std::vector<double>(numOfThirdHiddenLayerNeurons, 0.0);

		//Before ReLu
		firstHiddenLayerBeforeReLu = std::vector<double>(numOfFirstHiddenLayerNeurons, 0.0);
		secondHiddenLayerBeforeReLu = std::vector<double>(numOfSecondHiddenLayerNeurons, 0.0);
		thirdHiddenLayerBeforeReLu = std::vector<double>(numOfThirdHiddenLayerNeurons, 0.0);
		outputLayerBeforeSoftMax = std::vector<double>(numOfOutputLayerNeurons, 0.0);


		//Layers errors
		fullyconnectedLayerErrors = std::vector<double>(numOfFullyConnectedLayerNeurons, 0.0);
		outputLayerErrors = std::vector<double>(numOfOutputLayerNeurons, 0.0);
		firstHiddenLayerErrors = std::vector<double>(numOfFirstHiddenLayerNeurons, 0.0);
		secondHiddenLayerErrors = std::vector<double>(numOfSecondHiddenLayerNeurons, 0.0);
		thirdHiddenLayerErrors = std::vector<double>(numOfThirdHiddenLayerNeurons, 0.0);

		//Weights
		fullConToFirstHiddenWeights = std::vector<std::vector<double>>(numOfFullyConnectedLayerNeurons, std::vector<double>(numOfFirstHiddenLayerNeurons, 0.0));
		firstToSecondHiddenWeights = std::vector<std::vector<double>>(numOfFirstHiddenLayerNeurons, std::vector<double>(numOfSecondHiddenLayerNeurons, 0.0));
		secondToThirdHiddenWeights = std::vector<std::vector<double>>(numOfSecondHiddenLayerNeurons, std::vector<double>(numOfThirdHiddenLayerNeurons, 0.0));
		thirdHiddenToOutputWeights = std::vector<std::vector<double>>(numOfThirdHiddenLayerNeurons, std::vector<double>(numOfOutputLayerNeurons, 0.0));

		//Velcity
		fullConToFirstHiddenVelocity = std::vector<std::vector<double>>(numOfFullyConnectedLayerNeurons, std::vector<double>(numOfFirstHiddenLayerNeurons, 0.0));
		firstToSecondHiddenVelocity = std::vector<std::vector<double>>(numOfFirstHiddenLayerNeurons, std::vector<double>(numOfSecondHiddenLayerNeurons, 0.0));
		secondToThirdHiddenVelocity = std::vector<std::vector<double>>(numOfSecondHiddenLayerNeurons, std::vector<double>(numOfThirdHiddenLayerNeurons, 0.0));
		thirdHiddenToOutputVelocity = std::vector<std::vector<double>>(numOfThirdHiddenLayerNeurons, std::vector<double>(numOfOutputLayerNeurons, 0.0));

		//Bias
		firstHiddenLayerBias = std::vector<double>(numOfFirstHiddenLayerNeurons, 0.001);
		secondHiddenLayerBias = std::vector<double>(numOfSecondHiddenLayerNeurons, 0.001);
		thirdHiddenLayerBias = std::vector<double>(numOfThirdHiddenLayerNeurons, 0.001);
		outputLayerBias = std::vector<double>(numOfOutputLayerNeurons, 0.001);


		InitMatrixWeights(fullConToFirstHiddenWeights);
		InitMatrixWeights(firstToSecondHiddenWeights);
		InitMatrixWeights(secondToThirdHiddenWeights);
		InitMatrixWeights(thirdHiddenToOutputWeights);
}

Classifier::~Classifier()
{

}

std::vector<std::vector<double>> Classifier::TransponMatrix(const std::vector<std::vector<double>>& matrix) {

		std::vector<std::vector<double>> temp(matrix[0].size(), std::vector<double>(matrix.size(), 0.0));
		for (int i = 0; i < matrix.size(); i++) {
			for (int j = 0; j < matrix[i].size(); j++) {
				temp[j][i] = matrix[i][j];
			}
		}
		return temp;
	}

	void Classifier::InputErrorsForBackprop() {
		std::fill(fullyconnectedLayerErrors.begin(), fullyconnectedLayerErrors.end(), 0.0);
		for (int i = 0; i < fullyConnectedLayer.size(); i++) {
			double errSum = 0.0;
			for (int j = 0; j < firstHiddenLayerErrors.size(); j++) {
				errSum += (firstHiddenLayerErrors[j] * fullConToFirstHiddenWeights[i][j]);
			}
			fullyconnectedLayerErrors[i] = errSum;
		}
	}

	double Classifier::Weight(const double& leftBoard, const double& rightBoard) {
		static std::random_device rd;
		static std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(leftBoard, rightBoard);
		return dis(gen);
	}

	void Classifier::InitMatrixWeights(std::vector < std::vector < double>>& matrix) {
		double bound = std::sqrt(6.0 / matrix.size());
		for (int i = 0; i < matrix.size(); i++) {
			for (int j = 0; j < matrix[0].size(); j++) {
				matrix[i][j] = Weight(-bound, bound);
			}
		}
	}

	double Classifier::VecSum(const std::vector<double>& vec) {
		double sum = 0.0;
		for (const double& n : vec) {
			sum += n;
		}
		return sum;
	}

	double Classifier::OutLayerSum() {
		double res = 0.0;

		for (const double& n : outputLayer) {
			res += std::exp(n);
		}

		return res;
	}

	std::vector<double> Classifier::HiddenLayersCalcul(const std::vector<double>& layer, const std::vector<std::vector<double>>& layerWeights
		, std::vector<double>& layerBeforeReLu, const std::vector<double>& layerBias) {
		std::vector<double> res(layerWeights[0].size(), 0.0);

		for (int i = 0; i < layerWeights.size(); i++) {
			for (int j = 0; j < layerWeights[i].size(); j++) {
				res[j] += (layer[i] * layerWeights[i][j]);
			}
		}
		for (int i = 0; i < res.size(); i++) {
			res[i] += layerBias[i];
			layerBeforeReLu[i] = res[i];
			res[i] = (this->*ActivationFunction)(res[i]);
		}

		return res;
	}

	void Classifier::Classification() {
		ActivationFunction = &Classifier::ReLu;

		firstHiddenLayer = HiddenLayersCalcul(fullyConnectedLayer, fullConToFirstHiddenWeights, firstHiddenLayerBeforeReLu, firstHiddenLayerBias);

		secondHiddenLayer = HiddenLayersCalcul(firstHiddenLayer, firstToSecondHiddenWeights, secondHiddenLayerBeforeReLu, secondHiddenLayerBias);

		thirdHiddenLayer = HiddenLayersCalcul(secondHiddenLayer, secondToThirdHiddenWeights, thirdHiddenLayerBeforeReLu, thirdHiddenLayerBias);

		ActivationFunction = &Classifier::Linaer;
		//std::vector<double> dummyVec;
		outputLayer = HiddenLayersCalcul(thirdHiddenLayer, thirdHiddenToOutputWeights, outputLayerBeforeSoftMax, outputLayerBias);
		ApplySoftMax();
	}

	void Classifier::ApplySoftMax() {
		double maxOut = *std::max_element(outputLayer.begin(), outputLayer.end());
		double minOut = *std::min_element(outputLayer.begin(), outputLayer.end());
		std::cout << std::scientific << std::setprecision(10);

		std::cout << "Logs min\t" << minOut << std::endl;
		std::cout << "Logs max\t" << maxOut << std::endl;
		std::cout << "Logs dif\t" << maxOut - minOut << std::endl;

		for (size_t i = 0; i < outputLayer.size(); ++i) {
        if (!std::isfinite(outputLayer[i])) {
            throw std::string("ApplySoftMax: outputLayer already contains NaN/INF at index " 
                              + std::to_string(i) + " BEFORE calculation!");
        	}
    	}
		double res = 0.0;
		double maxOutput  = *std::max_element(outputLayer.begin(), outputLayer.end());
		for (const double& n : outputLayer) {
			res += std::exp(n - maxOutput);
		}

		for (double& n : outputLayer) {
			n = SoftMax(n - maxOutput,res);
		}
	}

	double Classifier::Linaer(double res) { return res; }

	int Classifier::FindCorrectOutNeuro() {
		auto it = std::max_element(outputLayer.begin(), outputLayer.end());
		auto res = outputLayer.begin();
		for (int i = 0; i < outputLayer.size(); i++) {
			if ((res + i) == it) {
				return i;
			}
		}
		return -1;
	}

	void Classifier::ShowLayer(int i) {
		auto show = [](const std::vector<double>& layer) {
			for (auto n : layer) {
				std::cout << n << "\t";
			}
			std::cout << std::endl;
			};

		switch (i) {
		case 1: show(fullyConnectedLayer); break;
		case 2: show(firstHiddenLayer); break;
		case 3: show(secondHiddenLayer); break;
		case 4: show(thirdHiddenLayer); break;
		case 5: show(outputLayer); break;
		default:
			std::cout << "No such layer" << std::endl;
		}
	}

	//Learning
	void Classifier::OutputLayerErrorCalcu(int correctNeuronIndex) {
		std::vector<double> targets(outputLayerErrors.size(), 0.0);
		targets[correctNeuronIndex] = 1.0;
		for (int i = 0; i < outputLayerErrors.size(); i++) {
			outputLayerErrors[i] = (outputLayer[i]  - targets[i]);
		}
	}

	void Classifier::HiddenLayerErrorCalcu(const std::vector<double>& nextLayErrors, const std::vector<std::vector<double>>& nextlayerWeights,
		const std::vector<double>& beforeRelu, std::vector<double>& currentLayErrors) {

		for (int i = 0; i < currentLayErrors.size(); i++) {
			double errSum = 0.0;
			for (int j = 0; j < nextLayErrors.size(); j++) {
				errSum += (nextLayErrors[j] * nextlayerWeights[i][j]);
			}
			currentLayErrors[i] = errSum * DirectiveReLu(beforeRelu[i]);
		}
	}

	void Classifier::Velocity(const std::vector<double>& currentLay, const std::vector<double> nextLayError, std::vector<std::vector<double>>& velocity) {
		double maxGradient = 0.0;
		for (int i = 0; i < currentLay.size(); i++) {
			for (int j = 0; j < nextLayError.size(); j++) {
				double gradient = currentLay[i] * nextLayError[j];
				maxGradient = std::max(maxGradient,std::abs(gradient));
				velocity[i][j] = (inertia * velocity[i][j]) + (LR * gradient);
			}
		}
		std::cout << "Max gradient:\t" << maxGradient << std::endl;
	}

	void Classifier::UpdateLayerWieghts(const std::vector<std::vector<double>>& velocity, std::vector<std::vector<double>>& layerWeights) {
		for (int i = 0; i < layerWeights.size(); i++) {
			for (int j = 0; j < layerWeights[i].size(); j++) {
				layerWeights[i][j] = layerWeights[i][j] - velocity[i][j];
			}
		}
	}

	void Classifier::UpdateBias(std::vector<double>& layerBias, const std::vector<double>& layerErrors) {
		for (int i = 0; i < layerBias.size(); i++) {
			layerBias[i] = layerBias[i] - (LR * layerErrors[i]);
		}
	}

	void Classifier::ShowLayerOutWieght(const std::vector<std::vector<double>>& oldWeights) {
		for (int i = 0; i < secondToThirdHiddenWeights.size(); i++) {
			for (int j = 0; j < secondToThirdHiddenWeights[i].size(); j++) {

				std::cout << "Weight new: " << secondToThirdHiddenWeights[i][j] << "\tOld weight: " << oldWeights[i][j] << std::endl << "Difference: " << secondToThirdHiddenWeights[i][j] - oldWeights[i][j] << std::endl;
			}
		}
	}

	void Classifier::LearningClassifier(int correctNeuronIndex) {
		std::fill(outputLayerErrors.begin(), outputLayerErrors.end(), 0.0);
    	std::fill(thirdHiddenLayerErrors.begin(), thirdHiddenLayerErrors.end(), 0.0);
    	std::fill(secondHiddenLayerErrors.begin(), secondHiddenLayerErrors.end(), 0.0);
    	std::fill(firstHiddenLayerErrors.begin(), firstHiddenLayerErrors.end(), 0.0);
    	std::fill(fullyconnectedLayerErrors.begin(), fullyconnectedLayerErrors.end(), 0.0);

		std::vector<std::vector<double>> oldWeights = secondToThirdHiddenWeights;


		 auto clip_gradients = [](std::vector<double>& errors) {
    		const double limit = 0.5; // Оптимальный порог для весов
    			for (double& err : errors) {
        		if (err > limit) err = limit;
        		else if (err < -limit) err = -limit;
    			}
		};

		OutputLayerErrorCalcu(correctNeuronIndex);
		clip_gradients(outputLayerErrors);
		if (HasNaN(outputLayerErrors)) {
			throw std::string("Invalid in outputLayerErrors");
    	}
		HiddenLayerErrorCalcu(outputLayerErrors, thirdHiddenToOutputWeights, thirdHiddenLayerBeforeReLu, thirdHiddenLayerErrors);
		clip_gradients(thirdHiddenLayerErrors);
		if (HasNaN(thirdHiddenLayerErrors)) {
			throw std::string("Invalid in thirdHiddenLayerErrors");
    	}
		HiddenLayerErrorCalcu(thirdHiddenLayerErrors, secondToThirdHiddenWeights, secondHiddenLayerBeforeReLu, secondHiddenLayerErrors);
		clip_gradients(secondHiddenLayerErrors);
		if (HasNaN(secondHiddenLayerErrors)) {
			throw std::string("Invalid in secondHiddenLayerErrors");
    	}
		HiddenLayerErrorCalcu(secondHiddenLayerErrors, firstToSecondHiddenWeights, firstHiddenLayerBeforeReLu, firstHiddenLayerErrors);
		clip_gradients(firstHiddenLayerErrors);
		if (HasNaN(firstHiddenLayerErrors)) {
			throw std::string("Invalid in firstHiddenLayerErrors");
    	}

		InputErrorsForBackprop();
		clip_gradients(fullyconnectedLayerErrors);
		if (HasNaN(fullyconnectedLayerErrors)) {
			throw std::string("Invalid in fullyconnectedLayerErrors");
    	}

    	
		Velocity(thirdHiddenLayer, outputLayerErrors, thirdHiddenToOutputVelocity);
		Velocity(secondHiddenLayer, thirdHiddenLayerErrors, secondToThirdHiddenVelocity);
		Velocity(firstHiddenLayer, secondHiddenLayerErrors, firstToSecondHiddenVelocity);
		Velocity(fullyConnectedLayer, firstHiddenLayerErrors, fullConToFirstHiddenVelocity);

		UpdateLayerWieghts(thirdHiddenToOutputVelocity, thirdHiddenToOutputWeights);
		UpdateLayerWieghts(secondToThirdHiddenVelocity, secondToThirdHiddenWeights);
		UpdateLayerWieghts(firstToSecondHiddenVelocity, firstToSecondHiddenWeights);
		UpdateLayerWieghts(fullConToFirstHiddenVelocity, fullConToFirstHiddenWeights);

		UpdateBias(outputLayerBias, outputLayerErrors);
		UpdateBias(thirdHiddenLayerBias, thirdHiddenLayerErrors);
		UpdateBias(secondHiddenLayerBias, secondHiddenLayerErrors);
		UpdateBias(firstHiddenLayerBias, firstHiddenLayerErrors);
	}

	void Classifier::ShowErrors(const std::vector<double>& error) {
		for (double n : error) {
			std::cout << "Error: " << std::setprecision(10) << n << std::endl;
		}
	}

	double Classifier::GetError(int correctNeuron) const{

		const double epsilon = 1e-15;

		double propability = outputLayer[correctNeuron];

		//std::cout << "Correct propability:\t" << std::scientific << propability << std::endl;;

		//std::cout << "Max propability:\t" << *std::max_element(outputLayer.begin(), outputLayer.end()) << std::endl;

		//std::cout << "Sum propability:\t" << std::accumulate(outputLayer.begin(), outputLayer.end(),0.0);

		propability = std::max(outputLayer[correctNeuron],epsilon);
    	return -std::log(propability);
	}

	void Classifier::ZeroOutputBias() {
		std::fill(outputLayerBias.begin(), outputLayerBias.end(), 0.0);
	}
		

	//Activation Functions

	double Classifier::LeakyReLu(double res) {
		return (res > 0.0) ? res : res * 0.01;
	}

	double Classifier::DirectiveLeakyReLu(double res) {
		return (res > 0.0) ? 1.0 : 0.01;
	}

	double Classifier::ReLu(double res) {
		return (res > 0.0) ? res : 0.0;
	}

	double Classifier::DirectiveReLu(double res) {
		//if (std::isnan(res)) return 0.0;
		return (res > 0.0) ? 1.0 : 0.0;
	}

	double Classifier::SoftMax(double res,double outputSum) {
		return std::exp(res) / outputSum;
	}


	bool Classifier::HasNaN(const std::vector<double>& vec) {
    	for (double n : vec) if (std::isnan(n) || std::isinf(n)) return true;
    		return false;
	}

	void Classifier::Reset() {
	
    
    std::fill(firstHiddenLayer.begin(), firstHiddenLayer.end(), 0.0);
    std::fill(secondHiddenLayer.begin(), secondHiddenLayer.end(), 0.0);
    std::fill(thirdHiddenLayer.begin(), thirdHiddenLayer.end(), 0.0);
    std::fill(outputLayer.begin(), outputLayer.end(), 0.0);

 
    std::fill(firstHiddenLayerBeforeReLu.begin(), firstHiddenLayerBeforeReLu.end(), 0.0);
    std::fill(secondHiddenLayerBeforeReLu.begin(), secondHiddenLayerBeforeReLu.end(), 0.0);
    std::fill(thirdHiddenLayerBeforeReLu.begin(), thirdHiddenLayerBeforeReLu.end(), 0.0);
    std::fill(outputLayerBeforeSoftMax.begin(), outputLayerBeforeSoftMax.end(), 0.0);

   
    std::fill(outputLayerErrors.begin(), outputLayerErrors.end(), 0.0);
    std::fill(fullyconnectedLayerErrors.begin(), fullyconnectedLayerErrors.end(), 0.0);
    std::fill(firstHiddenLayerErrors.begin(), firstHiddenLayerErrors.end(), 0.0);
    std::fill(secondHiddenLayerErrors.begin(), secondHiddenLayerErrors.end(), 0.0);
    std::fill(thirdHiddenLayerErrors.begin(), thirdHiddenLayerErrors.end(), 0.0);
	}

	//Setters & Getters

	void Classifier::SetFullyConnectedLayer(const std::vector<double>& vec) {
		double maxIn = *std::max_element(vec.begin(), vec.end());
		std::cout << "Max data from CNN:\t" << maxIn << std::endl;
		this->fullyConnectedLayer = vec;
	}

	std::vector<double> Classifier::GetOutLayer() const {
		return this->outputLayer;
	}

	std::vector<double> Classifier::GetInputErrors() const {
		return this->fullyconnectedLayerErrors;
	}