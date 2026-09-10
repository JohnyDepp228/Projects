#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


void InitMatrix(std::vector < std::vector < double>>& matrix) {
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			matrix[i][j] = j;
		}
	}
}


struct Filter {
	std::vector<double> filter;


	void SetFilter(const std::vector<double>& filter) {
		this->filter = filter;
	}

	std::vector<double> GetFilter() const {
		return this->filter;
	}
};

struct MapOfSigns {
	std::vector<double> mapOfSigns;


	void SetMapOfSigns(const std::vector<double>& mapOfSigns) {
		this->mapOfSigns.resize(mapOfSigns.size());
		this->mapOfSigns = mapOfSigns;
	}

	std::vector<double> GetMapOfSigns() const {
		return this->mapOfSigns;
	}

};

struct Channel {
	MapOfSigns* maps = nullptr;
	Filter* filters = nullptr;
	int amount = 0;
	int filterHeight = 3;
	int filterWidth = 3;
	int imageHeight = 840;
	int imageWidth = 840;

	void SetAmount(int amount) {
		maps = new MapOfSigns[amount];
		for (int i = 0; i < amount; i++) {
			maps[i].mapOfSigns.resize(imageHeight * imageWidth);
		}
		filters = new Filter[amount];
		for (int i = 0; i < amount; i++) {
			filters[i].filter.resize(filterHeight * filterWidth);
			InitFilterWeight(filters[i].filter);
		}
		this->amount = amount;
	}

	void InitFilterWeight(std::vector<double>& filter) {
		for (auto& n : filter) {
			n = Weight(-0.1, 0.1);
		}

	}

	double Weight(const double& leftBoard, const double& rightBoard) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(leftBoard, rightBoard);
		return dis(gen);
	}

	std::vector<double> ChannelSum() {
		std::vector<double> t = maps[0].mapOfSigns;
		for (int i = 1; i < amount - 1; i++) {
			for (int j = 0; j < maps[i].mapOfSigns.size(); j++)
				maps[i].mapOfSigns[j] += maps[i + 1].mapOfSigns[j];
		}
		return t;
	}


	void DoubleConvMaps(double bias, const std::vector<double>& vec) {
		int width = std::sqrt(vec.size());
		int height = std::sqrt(vec.size());
		int tHeight = height / 3;
		int tWidth = width / 3;
		std::vector<double> t(tHeight * tWidth);
		for (int fIdx = 0; fIdx < amount; fIdx++) {
			int mapIdx = 0;
			for (int i = 0; i <= height - 3; i += 3) {
				for (int j = 0; j <= width - 3; j += 3) {
					int filter = 0;
					double sum = 0.0;
					int idx = i * width + j;
					for (int kx = 0; kx < 3; kx++) {
						for (int ky = 0; ky < 3; ky++) {
							int newIdx = idx + ((kx * width) + ky);
							sum += vec[newIdx] * filters[fIdx].filter[filter++];
						}
					}
					sum += bias;
					t[mapIdx++] = ReLu(sum);
				}
			}
			mapIdx = 0;
			for (int i = 0; i <= tHeight - 3; i += 3) {
				for (int j = 0; j <= tWidth - 3; j += 3) {
					int filter = 0;
					double sum = 0.0;
					int idx = i * tWidth + j;
					for (int kx = 0; kx < 3; kx++) {
						for (int ky = 0; ky < 3; ky++) {
							int newIdx = idx + ((kx * tWidth) + ky);
							sum += t[newIdx] * filters[fIdx].filter[filter++];
						}
					}
					sum += bias;
					maps[fIdx].mapOfSigns[mapIdx++] = ReLu(sum);
				}
			}
		}
	}

	void Forward(double bias, const std::vector<double>& vec) {
		DoubleConvMaps(bias, vec);
		Pooling();
		//CleanZeroFromMap();
	}

	void RGBForward(double bias, const std::vector<double>& R,
		const std::vector<double>& G, const std::vector<double>& B) {
		CalculRGBMaps(bias, R, G, B);
		Pooling();
	}

	void CalculRGBMaps(double bias, const std::vector<double>& R,
		const std::vector<double>& G, const std::vector<double>& B) {
		ResetMaps();
		int mapIndex = 0;
		int stride = 1;
		int height = std::sqrt(R.size());
		int width = std::sqrt(R.size());

		int tHeight = height - filterHeight + 1;
		int tWidth = width - filterWidth + 1;
		std::vector<double> t(tHeight * tWidth);
		for (int q = 0; q < amount; q++) {
			int mapIndex = 0;

			for (int i = 0; i <= height - filterHeight; i += stride) {
				for (int j = 0; j <= width - filterWidth; j += stride) {
					double sum = 0.0;

					int filIdx = 0;

					for (int kx = 0; kx < filterHeight; kx++) {
						for (int ky = 0; ky < filterWidth; ky++) {
							int idx = (i + kx) * width + (ky + j);

							sum += R[idx] * filters[q].filter[filIdx];
							sum += G[idx] * filters[q].filter[filIdx];
							sum += B[idx] * filters[q].filter[filIdx];
							filIdx++;
						}
					}

					sum += bias;
					t[mapIndex++] = ReLu(sum);
				}
			}
			mapIndex = 0;
			for (int i = 0; i <= tHeight - 3; i += 3) {
				for (int j = 0; j <= tWidth - 3; j += 3) {
					int filter = 0;
					double sum = 0.0;
					int idx = i * tWidth + j;
					for (int kx = 0; kx < 3; kx++) {
						for (int ky = 0; ky < 3; ky++) {
							int newIdx = idx + ((kx * tWidth) + ky);
							sum += t[newIdx] * filters[q].filter[filter++];
						}
					}
					sum += bias;
					maps[q].mapOfSigns[mapIndex++] = ReLu(sum);
				}
			}
		}
	}

	void ResetMaps() {
		for (int i = 0; i < amount; i++) {
			std::fill(maps[i].mapOfSigns.begin(), maps[i].mapOfSigns.end(), 0.0);
		}
	}

	void Pooling() {
		int height = std::sqrt(maps[0].mapOfSigns.size());
		int width = std::sqrt(maps[0].mapOfSigns.size());
		std::vector<double> temp((maps[0].mapOfSigns.size() / 4), 0.0);
		for (int i = 0; i < amount; i++) {
			std::vector<double> temp(((height / 2) * (width / 2)), 0.0);
			int tIdx = 0;
			for (int x = 0; x < height; x += 2) {
				for (int y = 0; y < width; y += 2) {
					int idx = x * width + y;
					double t1 = std::max(maps[i].mapOfSigns[idx], maps[i].mapOfSigns[idx + 1]);
					double t2 = std::max(maps[i].mapOfSigns[idx + width], maps[i].mapOfSigns[idx + width + 1]);
					double res = std::max(t1, t2);
					temp[tIdx] = res;
					tIdx++;
				}
			}
			maps[i].SetMapOfSigns(temp);
			temp.clear();
		}
	}

	void CleanZeroFromMap() {
		for (int i = 0; i < amount; i++) {
			maps[i].mapOfSigns.erase(std::remove(maps[i].mapOfSigns.begin(), maps[i].mapOfSigns.end(), 0.0), maps[i].mapOfSigns.end());
		}
	}

	~Channel() {
		delete[] maps;
		delete[] filters;
	}

	//Activate Functions

	double LeakyReLu(double res) {
		return res > 0.0 ? res : res * 0.01;
	}

	double DirectiveLeakyReLu(double res) {
		return res > 0.0 ? res : 0.01;
	}

	double ReLu(double res) {
		return res > 0.0 ? res : 0.0;
	}

	double DirectiveReLu(double res) {
		return res > 0.0 ? 1 : 0.0;
	}

	//Setter & Getters

	std::vector<double > GetMap(int index) const {
		return maps[index].mapOfSigns;
	}

	int GetNumOfFilters() const {
		return this->amount;
	}

};

class CNN {
private:
	std::vector<std::vector<double>> photoMatrix;
	double bias = 0.4;

	int filterHeight = 3;
	int filterWeight = 3;
	int poolingWindowHeight = 2;
	int poolingWindowWidth = 2;

	int imageHeight = 840;
	int imageWidth = 840;

	int numOfBlocks = 7;
	int numOfFiltersInBlock = 8;

	Channel* channels;

public:
	CNN() {
		channels = new Channel[numOfBlocks];
		for (int i = 0; i < numOfBlocks; i++) {
			channels[i].SetAmount(numOfFiltersInBlock);
			numOfFiltersInBlock *= 2;
		}

	}

	std::vector<std::vector<double>> LoadImage(std::string imagePath, int color) {
		int width = 0;
		int height = 0;
		int channels = 0;
		unsigned char* res = stbi_load(imagePath.c_str(), &width, &height, &channels, 3);

		std::vector<std::vector<double>> matrix(height, std::vector<double>(width, 0.0));

		if (res == NULL) {
			std::cout << "Error read image\t" << stbi_failure_reason() << std::endl;
			exit(1);
			return matrix;
		}

		for (int x = 0; x < height; x++) {
			for (int y = 0; y < width; y++) {
				int pixelIndex = (x * width + y) * 3;

				matrix[x][y] = (double)res[pixelIndex + color];
			}
		}

		stbi_image_free(res);

		return matrix;
	}

	void ShowVector(const std::vector<double>& vec) {
		for (const auto& n : vec) {
			std::cout << n << " ";
		}

	}

	void ShowMatrix(const std::vector < std::vector < double>>& matrix) {
		for (const auto& n : matrix) {
			ShowVector(n);
			std::cout << std::endl;
		}
	}

	double GMP(const std::vector < double>& vec) {

		return *max_element(vec.begin(), vec.end());
	}

	std::vector < std::vector < double>> VectorIntoMatrix(const std::vector<double>& vec) {
		int matHeight = std::sqrt((double)vec.size());
		int matWidth = std::sqrt((double)vec.size());
		std::vector < std::vector < double>> res(matHeight, std::vector<double>(matWidth, 0.0));
		if (matHeight == 0) {
			matHeight = 1;
		}
		int border = vec.size() / matHeight;
		int vecIndex = 0;
		for (int i = 0; i < matHeight; i++) {
			for (int j = 0; j < matWidth; j++) {
				res[i][j] = vec[vecIndex];
				vecIndex++;
			}
		}
		return res;
	}

	std::vector < double> MatrixIntoVector(const std::vector < std::vector < double>>& matrix) {
		int vecSize = matrix.size() * matrix[0].size();
		std::vector < double> res;
		for (const auto& row : matrix) {
			for (const auto& col : row) {
				res.push_back(col);
			}
		}
		return res;
	}

	void IncreaseMatrixeSize(std::vector < std::vector < double>>& matrix) {
		if (matrix.size() < imageHeight) {
			matrix.resize(imageHeight);
		}

		for (int i = 0; i < imageHeight; i++) {
			if (matrix[i].size() < imageWidth) {
				matrix[i].resize(imageWidth);
			}
		}
	}

	void Padding(std::vector < std::vector < double>>& matrix) {
		std::vector<double> vec(matrix[0].size() + 2, 0.0);
		for (auto& n : matrix) {
			n.insert(n.begin(), 0.0);
			n.push_back(0.0);
		}
		matrix.push_back(vec);
		matrix.insert(matrix.begin(), vec);
	}

	std::vector < std::vector < double>> BilinearInterpolation(std::vector < std::vector < double>>& matrix) {

		std::vector < std::vector < double>> res(imageHeight, std::vector<double>(imageWidth, 0.0));

		double decr—oefX = (double)matrix.size() / (double)imageHeight;
		double decr—oefY = (double)matrix[0].size() / (double)imageWidth;
		for (int x = 0; x < imageHeight; x++) {
			for (int y = 0; y < imageWidth; y++) {
				double scaleX = (x + 0.5) * decr—oefX - 0.5;
				double scaleY = (y + 0.5) * decr—oefY - 0.5;

				int x1 = std::min((int)scaleX, (int)matrix[0].size() - 1);
				int x2 = std::min((int)scaleX + 1, (int)matrix[0].size() - 1);
				int y1 = std::min((int)scaleY, (int)matrix.size() - 1);
				int y2 = std::min((int)scaleY + 1, (int)matrix.size() - 1);

				double q11 = matrix[y1][x1];
				double q21 = matrix[y1][x2];
				double q12 = matrix[y2][x1];
				double q22 = matrix[y2][x2];

				double xWeight = scaleX - (int)scaleX;
				double yWeight = scaleY - (int)scaleY;

				double w11 = (1 - xWeight) * (1 - yWeight);
				double w21 = xWeight * (1 - yWeight);
				double w12 = (1 - xWeight) * yWeight;
				double w22 = xWeight * yWeight;

				res[x][y] = q11 * w11 + q21 * w21 + q12 * w12 + q22 * w22;
			}
		}

		return res;

	}

	void ChangeMatrixSize(std::vector < std::vector < double>>& matrix) {
		if (matrix.size() < imageHeight || matrix[0].size() < imageWidth) {
			IncreaseMatrixeSize(matrix);
		}
		else if (matrix.size() > imageHeight || matrix[0].size() > imageWidth) {
			matrix = BilinearInterpolation(matrix);
		}
		Padding(matrix);
	}

	void ForwardRGB(const std::vector<double>& R, const std::vector<double>& G, const std::vector<double>& B) {
		channels[0].RGBForward(bias, R, G, B);
	}

	std::vector<double> Forward() {
		int size = GetNumOFBlocks();
		std::vector<double> t(numOfFiltersInBlock);
		for (int i = 1; i < size; i++) {
			channels[i].Forward(bias, channels[i - 1].ChannelSum());
		}
		size = channels[numOfBlocks - 1].GetNumOfFilters();

		for (int i = 0; i < size; i++) {
			t[i] = GMP(channels[numOfBlocks - 1].GetMap(i));
		}
		t.erase(std::remove(t.begin(), t.end(), 0.0), t.end());
		return t;
	}

	void NormalizeImage(std::vector<std::vector<double>>& matrix) {
		for (auto& n : matrix) {
			for (auto& j : n) {
				j = j / 255.0;
			}
		}
	}

	~CNN() {
		delete[] channels;
	}



	//Activation Functions

	double LeakyReLu(double res) {
		return res > 0.0 ? res : res * 0.01;
	}

	double DirectiveLeakyReLu(double res) {
		return res > 0.0 ? res : 0.01;
	}

	double ReLu(double res) {
		return res > 0.0 ? res : 0.0;
	}

	double DirectiveReLu(double res) {
		return res > 0.0 ? 1 : 0.0;
	}

	//Setter & Getters
	std::vector < std::vector < double>> GetPhotoMatrix() const {
		return photoMatrix;
	}

	void SetImageMatrix(const std::vector<std::vector<double>>& matrix) {
		this->photoMatrix = matrix;
	}

	int GetNumOFBlocks() const {
		return this->numOfBlocks;
	}

	std::vector<double> GetMap(int channelInx) const {
		return channels[channelInx].GetMap(0);
	}

};


class Classifier {
private:
	int numOfFullyConnectedLayerNeurons = 512;
	int numOfFirstHiddenLayerNeurons = 460;
	int numOfSecondHiddenLayerNeurons = 230;
	int numOfThirdHiddenLayerNeurons = 130;
	int numOfOutputLayerNeurons = 100;
	double inertia = 0.01;
	double LR = 0.002;

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
public:
	Classifier() {
		//Layers
		outputLayer = std::vector<double>(numOfOutputLayerNeurons, 0.0);
		firstHiddenLayer = std::vector<double>(numOfFirstHiddenLayerNeurons, 0.0);
		secondHiddenLayer = std::vector<double>(numOfSecondHiddenLayerNeurons, 0.0);
		thirdHiddenLayer = std::vector<double>(numOfThirdHiddenLayerNeurons, 0.0);

		//Before ReLu
		firstHiddenLayerBeforeReLu = std::vector<double>(numOfFirstHiddenLayerNeurons, 0.0);
		secondHiddenLayerBeforeReLu = std::vector<double>(numOfSecondHiddenLayerNeurons, 0.0);
		thirdHiddenLayerBeforeReLu = std::vector<double>(numOfThirdHiddenLayerNeurons, 0.0);


		//Layers errors
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
		firstHiddenLayerBias = std::vector<double>(numOfFirstHiddenLayerNeurons, 0.0);
		secondHiddenLayerBias = std::vector<double>(numOfSecondHiddenLayerNeurons, 0.0);
		thirdHiddenLayerBias = std::vector<double>(numOfThirdHiddenLayerNeurons, 0.0);
		outputLayerBias = std::vector<double>(numOfOutputLayerNeurons, 0.0);


		InitMatrixWeights(fullConToFirstHiddenWeights);
		InitMatrixWeights(firstToSecondHiddenWeights);
		InitMatrixWeights(secondToThirdHiddenWeights);
		InitMatrixWeights(thirdHiddenToOutputWeights);
	}

	double Weight(const double& leftBoard, const double& rightBoard) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(leftBoard, rightBoard);
		return dis(gen);
	}

	void InitMatrixWeights(std::vector < std::vector < double>>& matrix) {
		for (int i = 0; i < matrix.size(); i++) {
			for (int j = 0; j < matrix[0].size(); j++) {
				matrix[i][j] = Weight(-0.2, 0.2);
			}
		}
	}

	double VecSum(const std::vector<double>& vec) {
		double sum = 0.0;
		for (const double& n : vec) {
			sum += n;
		}
		return sum;
	}

	double OutLayerSum() {
		double res = 0.0;

		for (const double& n : outputLayer) {
			res += std::exp(n);
		}

		return res;
	}

	std::vector<double> HiddenLayersCalcul(const std::vector<double>& layer, const std::vector<std::vector<double>>& layerWeights
		, std::vector<double>& layerBeforeReLu, const std::vector<double>& layerBias) {
		std::vector<double> res(layerWeights[0].size(), 0.0);

		for (int i = 0; i < layerWeights.size(); i++) {
			for (int j = 0; j < layerWeights[i].size(); j++) {
				res[j] += (layer[j] * layerWeights[i][j]);
			}
		}
		layerBeforeReLu.resize(res.size());
		for (int i = 0; i < res.size(); i++) {
			res[i] += layerBias[i];
			layerBeforeReLu[i] = res[i];
			res[i] = (this->*ActivationFunction)(res[i]);
		}

		return res;
	}

	void Classification() {
		ActivationFunction = &Classifier::ReLu;

		firstHiddenLayer = HiddenLayersCalcul(fullyConnectedLayer, fullConToFirstHiddenWeights, firstHiddenLayerErrors, firstHiddenLayerBias);

		secondHiddenLayer = HiddenLayersCalcul(firstHiddenLayer, firstToSecondHiddenWeights, secondHiddenLayerErrors, secondHiddenLayerBias);

		thirdHiddenLayer = HiddenLayersCalcul(secondHiddenLayer, secondToThirdHiddenWeights, thirdHiddenLayerErrors, thirdHiddenLayerBias);

		ActivationFunction = &Classifier::Linaer;
		std::vector<double> dummyVec;
		outputLayer = HiddenLayersCalcul(thirdHiddenLayer, thirdHiddenToOutputWeights, dummyVec, outputLayerBias);
		ApplySoftMax();
	}

	void ApplySoftMax() {
		for (double& n : outputLayer) {
			n = SoftMax(n);
		}
	}

	double Linaer(double res) { return res; }

	int FindCorrectOutNeuro() {
		auto it = std::max_element(outputLayer.begin(), outputLayer.end());
		auto res = outputLayer.begin();
		for (int i = 0; i < outputLayer.size(); i++) {
			if ((res + i) == it) {
				return i;
			}
		}
		return -1;
	}

	void ShowLayer(int i) {
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
	void OutputLayerErrorCalcu(const std::vector<double>& targets) {
		for (int i = 0; i < outputLayerErrors.size(); i++) {
			outputLayerErrors[i] = (targets[i] - outputLayerErrors[i]);
		}
	}

	void HiddenLayerErrorCalcu(const std::vector<double>& nextLayErrors, const std::vector<std::vector<double>>& nextlayerWeights,
		const std::vector<double>& beforeRelu, std::vector<double>& currentLayErrors) {

		for (int i = 0; i < currentLayErrors.size(); i++) {
			double errSum = 0.0;
			for (int j = 0; j < nextLayErrors.size(); j++) {
				errSum += (nextLayErrors[j] * nextlayerWeights[i][j]);
			}
			currentLayErrors[i] = errSum * DirectiveReLu(beforeRelu[i]);
		}
	}

	void Velocity(const std::vector<double>& currentLay, const std::vector<double> nextLayError, std::vector<std::vector<double>>& velocity) {
		for (int i = 0; i < currentLay.size(); i++) {
			for (int j = 0; j < nextLayError.size(); j++) {
				double gradient = currentLay[i] * nextLayError[j];
				velocity[i][j] = (inertia * velocity[i][j]) + (LR * gradient);
			}
		}
	}

	void UpdateLayerWieghts(const std::vector<std::vector<double>>& velocity, std::vector<std::vector<double>>& layerWeights) {
		for (int i = 0; i < layerWeights.size(); i++) {
			for (int j = 0; j < layerWeights[i].size(); j++) {
				layerWeights[i][j] = layerWeights[i][j] + velocity[i][j];
			}
		}
	}

	void UpdateBias(std::vector<double>& layerBias, const std::vector<double>& layerErrors) {
		for (int i = 0; i < layerBias.size(); i++) {
			layerBias[i] = layerBias[i] + (LR * layerErrors[i]);
		}
	}

	void Learning(const std::vector<double>& targets) {
		OutputLayerErrorCalcu(targets);
		HiddenLayerErrorCalcu(outputLayerErrors, thirdHiddenToOutputWeights, thirdHiddenLayerBeforeReLu, thirdHiddenLayerErrors);
		HiddenLayerErrorCalcu(thirdHiddenLayerErrors, secondToThirdHiddenWeights, secondHiddenLayerBeforeReLu, secondHiddenLayerErrors);
		HiddenLayerErrorCalcu(secondHiddenLayerErrors, firstToSecondHiddenWeights, firstHiddenLayerBeforeReLu, firstHiddenLayerErrors);



		ShowErrors(outputLayerErrors);

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

	void ShowErrors(const std::vector<double>& error) {
		for (double n : error) {
			std::cout << "Error: " << std::setprecision(10) << n << std::endl;
		}
	}

	//Activation Functions

	double LeakyReLu(double res) {
		return res > 0.0 ? res : res * 0.01;
	}

	double DirectiveLeakyReLu(double res) {
		return res > 0.0 ? res : 0.01;
	}

	double ReLu(double res) {
		return res > 0.0 ? res : 0.0;
	}

	double DirectiveReLu(double res) {
		return (res > 0.0) ? 1.0 : 0.0;
	}

	double SoftMax(double res) {
		return std::exp(res) / Classifier::OutLayerSum();
	}

	//Setters & Getters

	void SetFullyConnectedLayer(const std::vector<double>& vec) {
		this->fullyConnectedLayer = vec;
	}

	std::vector<double> GetOutLayer() const {
		return this->outputLayer;
	}
};

int Predict(std::string path) {
	std::vector<double> target(100, 0.0);
	target[98] = 1.0;
	CNN c;
	Classifier cl;
	std::vector < std::vector < double>> matrixR = c.LoadImage(path, 0);
	std::vector < std::vector < double>> matrixG = c.LoadImage(path, 1);
	std::vector < std::vector < double>> matrixB = c.LoadImage(path, 2);
	c.NormalizeImage(matrixR);
	c.NormalizeImage(matrixG);
	c.NormalizeImage(matrixB);
	c.ChangeMatrixSize(matrixR);
	c.ChangeMatrixSize(matrixG);
	c.ChangeMatrixSize(matrixB);

	std::vector<double> R = c.MatrixIntoVector(matrixR);
	std::vector<double> G = c.MatrixIntoVector(matrixG);
	std::vector<double> B = c.MatrixIntoVector(matrixB);

	c.ForwardRGB(R, G, B);

	std::vector<double> res = c.Forward();

	cl.SetFullyConnectedLayer(res);
	cl.Classification();
	std::cout << "Learning..." << std::endl;
	cl.Learning(target);
	std::cout << "Done" << std::endl;
	return cl.FindCorrectOutNeuro();
}


int main()
{
	auto start = std::chrono::high_resolution_clock::now();

	std::string path = "C:/Users/Boss/Desktop/Ò ÙÎÂ¯ÍË/2024_12_13 FOTO/13_12_0954.jpg";
	std::string path2 = "C:/Users/LordMegatron/Desktop/2.jpg";

	std::cout << Predict(path2) << std::endl;

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = end - start;

	std::cout << "Execution time: " << duration.count() << std::endl;
	system("pause");

	return 0;

}
