#include "convneurnet.h"

ConvNeurNet::ConvNeurNet()
{
    channels = new Channal[numOfBlocks];
		for (int i = 0; i < numOfBlocks; i++) {
			channels[i].SetAmount(numOfFiltersInBlock);

			numOfFiltersInBlock *= 2;
		}
}

ConvNeurNet::~ConvNeurNet()
{
    delete[] channels;
}


std::vector<std::vector<double>> ConvNeurNet::LoadImage(std::string imagePath, int color) {
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
	void ConvNeurNet::UpdateFiltersWeights(int channalIdx) {
		channels[channalIdx].UpdateFilterWeights();
	}

	void ConvNeurNet::CalculMatrixOfErrorsInChannel(int channalIdx) {
		channels[channalIdx].CalculMatrixOfErrors();
	}

	void ConvNeurNet::MarkingErrorsOnMap(const std::vector<double>& fullyconnectedLayerErrors) {
		int el = 0;
		for (int i = 0; i < fullyconnectedLayerErrors.size(); i++) {
			int index = channels[numOfBlocks - 1].GetMaxElementIndex(i);
			channels[numOfBlocks - 1].SetErrorMap(i, fullyconnectedLayerErrors[el], index);
			el++;
		}

	}

	void ConvNeurNet::ShowVector(const std::vector<double>& vec) {
		for (const auto& n : vec) {
			std::cout << n << " ";
		}

	}

	void ConvNeurNet::ShowMatrix(const std::vector < std::vector < double>>& matrix) {
		for (const auto& n : matrix) {
			ShowVector(n);
			std::cout << std::endl;
		}
	}

	double ConvNeurNet::GMP(const std::vector < double>& vec) {

		return *max_element(vec.begin(), vec.end());
	}

	double ConvNeurNet::GMP(const std::vector < double>& vec, int& maxElIndex) {
		auto it = *max_element(vec.begin(), vec.end());
		for (int i = 0; i < vec.size(); i++) {
			if (vec[i] == it) {
				maxElIndex = i;
				break;
			}
		}
		return it;
	}

	std::vector < std::vector < double>> ConvNeurNet::VectorIntoMatrix(const std::vector<double>& vec) {
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

	std::vector < double> ConvNeurNet::MatrixIntoVector(const std::vector < std::vector < double>>& matrix) {
		//int vecSize = matrix.size() * matrix[0].size();
		std::vector < double> res;
		for (const auto& row : matrix) {
			for (const auto& col : row) {
				res.push_back(col);
			}
		}
		return res;
	}

	void ConvNeurNet::IncreaseMatrixeSize(std::vector < std::vector < double>>& matrix) {
		if (matrix.size() < imageHeight) {
			matrix.resize(imageHeight);
		}

		for (int i = 0; i < imageHeight; i++) {
			if (matrix[i].size() < imageWidth) {
				matrix[i].resize(imageWidth);
			}
		}
	}

	void ConvNeurNet::Padding(std::vector < std::vector < double>>& matrix) {
		std::vector<double> vec(matrix[0].size() + 2, 0.0);
		for (auto& n : matrix) {
			n.insert(n.begin(), 0.0);
			n.push_back(0.0);
		}
		matrix.push_back(vec);
		matrix.insert(matrix.begin(), vec);
	}

	std::vector < std::vector < double>> ConvNeurNet::BilinearInterpolation(std::vector < std::vector < double>>& matrix) {

		std::vector < std::vector < double>> res(imageHeight, std::vector<double>(imageWidth, 0.0));

		double decrСoefX = (double)matrix.size() / (double)imageHeight;
		double decrСoefY = (double)matrix[0].size() / (double)imageWidth;
		for (int x = 0; x < imageHeight; x++) {
			for (int y = 0; y < imageWidth; y++) {
				double scaleX = (x + 0.5) * decrСoefX - 0.5;
				double scaleY = (y + 0.5) * decrСoefY - 0.5;

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

	void ConvNeurNet::ChangeMatrixSize(std::vector < std::vector < double>>& matrix) {
		if (matrix.size() < imageHeight || matrix[0].size() < imageWidth) {
			IncreaseMatrixeSize(matrix);
		}
		else if (matrix.size() > imageHeight || matrix[0].size() > imageWidth) {
			matrix = BilinearInterpolation(matrix);
		}
		Padding(matrix);
	}

	void ConvNeurNet::ForwardRGB(const std::vector<double>& R, const std::vector<double>& G, const std::vector<double>& B) {
		channels[0].RGBForward(bias, R, G, B);
	}

	std::vector<double> ConvNeurNet::Forward() {
		int size = GetNumOFBlocks();
		std::cout << "Size:" << size << std::endl;
		std::vector<double> t(numOfFiltersInBlock);
		for (int i = 1; i < size; i++) {
			std::cout << "Channels " << i << "forward done" << std::endl;
			channels[i].Forward(bias, channels[i - 1].ChannelSum());
		}
		std::cout << "Channels forward done" << std::endl;
		size = channels[numOfBlocks - 1].GetNumOfFilters();
		int maxElementIndex = 0;
		for (int i = 0; i < size; i++) {
			maxElementIndex = 0;
			t[i] = GMP(channels[numOfBlocks - 1].GetMap(i), maxElementIndex);
			channels[numOfBlocks - 1].SetMaxElementIndex(i, maxElementIndex);
		}
		t.erase(std::remove(t.begin(), t.end(), 0.0), t.end());
		return t;
	}

	void ConvNeurNet::NormalizeImage(std::vector<std::vector<double>>& matrix) {
		for (auto& n : matrix) {
			for (auto& j : n) {
				j = j / 255.0;
			}
		}
	}

	void ConvNeurNet::LearningConvLayers(const std::vector<double>& fullyconnectedLayerErrors) {
		std::cout << "Filter before" << std::endl;
		ShowVector(channels[numOfBlocks - 1].GetFilter(0));
		std::cout << "Learn" << std::endl;
		MarkingErrorsOnMap(fullyconnectedLayerErrors);
		CalculMatrixOfErrorsInChannel(numOfBlocks - 1);
		UpdateFiltersWeights(numOfBlocks - 1);
		std::cout << "Filter after" << std::endl;
		ShowVector(channels[numOfBlocks - 1].GetFilter(0));

	}

	//Activation Functions

	double ConvNeurNet::LeakyReLu(double res) {
		return res > 0.0 ? res : res * 0.01;
	}

	double ConvNeurNet::DirectiveLeakyReLu(double res) {
		return res > 0.0 ? res : 0.01;
	}

	double ConvNeurNet::ReLu(double res) {
		return res > 0.0 ? res : 0.0;
	}

	double ConvNeurNet::DirectiveReLu(double res) {
		return res > 0.0 ? 1 : 0.0;
	}

	double ConvNeurNet::GeLu(double res) {
		return 0.5 * res * (1 + std::tanh(std::sqrt(2 / std::numbers::pi) * (res + 0.044715 * std::pow(res, 3))));
	}

	double ConvNeurNet::DirectiveGeLu(double res) {
		double y = std::sqrt(2 / std::numbers::pi) * (res + 0.044715 * std::pow(res, 3));
		return 0.5 * (1 + tanh(y)) + 0.5 * res * (1 - std::pow(tanh(y), 2)) * std::sqrt(2 / std::numbers::pi) * (1 + 0.134145 * std::pow(res, 2));
	}

	//Setter & Getters
	std::vector < std::vector < double>> ConvNeurNet::GetPhotoMatrix() const {
		return photoMatrix;
	}

	void ConvNeurNet::SetImageMatrix(const std::vector<std::vector<double>>& matrix) {
		this->photoMatrix = matrix;
	}

	int ConvNeurNet::GetNumOFBlocks() const {
		return this->numOfBlocks;
	}

	std::vector<double> ConvNeurNet::GetMap(int channelInx) const {
		return channels[channelInx].GetMap(0);
	}