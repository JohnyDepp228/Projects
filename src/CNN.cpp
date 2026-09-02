#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <chrono>


void InitMatrix(std::vector < std::vector < double>>& matrix) {
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			matrix[i][j] = j;
		}
	}
}

struct Channel {
	std::vector<double> mapOfSigns;

	std::vector<double> filter;

	int numOfFilters = 0;

	int size = 0;

	void SetMap(const std::vector<double>& mapOfSigns, int size) {
		this->size = size;
		this->mapOfSigns = mapOfSigns;
	}

	std::vector<double> GetMap() const {
		return this->mapOfSigns;
	}

	void SetFilter(const std::vector<double>& filter) {
		this->filter = filter;
	}

	std::vector<double> GetFilter() const {
		return this->filter;
	}

	int GetMapSize() const {
		return size;
	}

};


class CNN {
private:
	std::vector<std::vector<double>> photoMatrix;
	double bias = 1.0;

	int filterHeight = 3;
	int filterWeight = 3;
	int poolingWindowHeight = 2;
	int poolingWindowWidth = 2;


	int imageHeight = 840;
	int imageWidth = 840;

	int numOfBlocks = 7;
	int numOfFiltersInBlock = 8;
	Channel** blocks;


public:
	CNN() {
		int filterSize = filterHeight * filterWeight;
		blocks = new Channel * [numOfBlocks];
		for (int i = 0; i < numOfBlocks; i++) {
			blocks[i] = new Channel[numOfFiltersInBlock];
			for (int j = 0; j < numOfFiltersInBlock; j++) {
				blocks[i][j].SetFilter(InitFilterWeight(filterSize));
			}
			blocks[i]->numOfFilters = numOfFiltersInBlock;

			numOfFiltersInBlock *= 2;
		}

	}

	std::vector<double> InitFilterWeight(const int& filterSize) {
		std::vector<double> filter(filterSize, 0);
		for (auto& n : filter) {
			n = Weight(-0.5, 0.5);
		}
		return filter;
	}

	double Weight(const double& leftBoard, const double& rightBoard) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(leftBoard, rightBoard);
		return dis(gen);
	}


	std::vector<double> GetSmallerMatrixFromMatrix(const int& startI, const int& startJ, const std::vector<std::vector<double>>& matrix,
		const int& smallMatrixHeight, const int& smallMatrixWidth) {
		double res = 0.0;
		std::vector<double> temp;
		for (int i = startI; i < startI + smallMatrixHeight; i++) {
			for (int j = startJ; j < startJ + smallMatrixWidth; j++) {
				if (i < matrix.size() && j < matrix[0].size()) {
					temp.push_back(matrix[i][j]);
				}
			}
		}
		return temp;
	}

	double —onvolutionOfOneMatrix(const std::vector<double>& MatrixFromPhoto, const int& channelIndex, const int& filterIndex) {
		double res = 0.0;
		std::vector<double> filter = blocks[channelIndex][filterIndex].GetFilter();
		for (int i = 0; i < MatrixFromPhoto.size(); i++) {
			res += MatrixFromPhoto[i] * filter[i];
		}
		res += bias;

		return ReLu(res);

	}

	double MaxPooling(const std::vector<double>& smallMapOfSigns) {
		if (smallMapOfSigns.empty()) {
			return 0.0;
		}
		else {
			return *max_element(smallMapOfSigns.begin(), smallMapOfSigns.end());
		}
	}

	double MinPooling(const std::vector<double>& smallMapOfSigns) {
		if (smallMapOfSigns.empty()) {
			return 0.0;
		}
		else {
			return *min_element(smallMapOfSigns.begin(), smallMapOfSigns.end());
		}
	}

	std::vector < std::vector < double>> FullConvolution(std::vector < std::vector < double>>& matrix, const int& biggerMatHeight, const int& biggerMatWiedth,
		const int& smallerMatHeight, const int& smallerMatWeight, int& newHeight, int& newWidth, const int& channelIndex, const int& filterIndex) {
		newHeight = (biggerMatHeight - smallerMatHeight) + 1;
		newWidth = (biggerMatWiedth - smallerMatWeight) + 1;
		std::vector < std::vector < double>> mapOfSigns((biggerMatHeight - smallerMatHeight) + 1, std::vector < double>((biggerMatWiedth - smallerMatWeight) + 1, 0));
		for (int i = 0; i <= biggerMatHeight - smallerMatHeight; i++) {
			for (int j = 0; j <= biggerMatWiedth - smallerMatWeight; j++) {
				mapOfSigns[i][j] = —onvolutionOfOneMatrix(GetSmallerMatrixFromMatrix(i, j, matrix, smallerMatHeight, smallerMatWeight), channelIndex, filterIndex);
			}
		}

		return mapOfSigns;
	}

	void BlokOfConvNPool(const int& channelIndex, const int& filterIndex, std::vector < std::vector < double>>& newFilter) {
		int newHeight = 0;
		int newWidth = 0;
		std::vector < std::vector < double>> conv1 = FullConvolution(newFilter, newFilter.size(), newFilter[0].size(), filterHeight, filterWeight, newHeight, newWidth, channelIndex, filterIndex);
		std::vector < std::vector < double>> conv2 = FullConvolution(conv1, conv1.size(), conv1[0].size(), filterHeight, filterWeight, newHeight, newWidth, channelIndex, filterIndex);
		newFilter = Pooling(conv2);
	}

	std::vector < std::vector < double>> MatrixForPooling(const int& stride, const std::vector < std::vector < double>>& mapOfSigns) {
		std::vector < double> res;
		std::vector < std::vector < double>> res2;
		int numOfmat = mapOfSigns.size() / stride;
		int x = 0;
		int y = 0;
		for (int h = 0; h < numOfmat; h++) {
			y = 0;
			for (int w = 0; w < numOfmat; w++) {
				for (int i = x; i < x + stride; i++) {
					for (int j = y; j < y + stride; j++) {
						res.push_back(mapOfSigns[i][j]);
					}
				}
				res2.push_back(res);
				res.clear();
				y += stride;
			}
			x += stride;
		}
		return res2;
	}

	std::vector < std::vector < double>> Pooling(const std::vector < std::vector < double>>& mapOfSigns) {
		int stride = 2;
		int sizeH = mapOfSigns.size() / stride;
		int sizeW = mapOfSigns.size() / stride;

		std::vector<double> res;
		std::vector < std::vector < double>> res2 = MatrixForPooling(stride, mapOfSigns);
		for (auto n : res2) {
			res.push_back(MaxPooling(n));
		}


		return VectorIntoMatrix(res);
	}

	std::vector < std::vector < double>> GetPhotoMatrix() const {
		return photoMatrix;
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

	double GAP(const std::vector < std::vector < double>>& matrix) {
		std::vector<double> temp;
		for (const auto& n : matrix) {
			temp.push_back(MaxPooling(n));
		}

		return *max_element(temp.begin(), temp.end());
	}

	std::vector < std::vector < double>> VectorIntoMatrix(const std::vector<double>& vec) {
		int matHeight = std::sqrt((double)vec.size());
		int matWidth = std::sqrt((double)vec.size());
		std::vector < std::vector < double>> res(matHeight, std::vector<double>(matWidth, 0));
		if (matHeight == 0) {
			matHeight = 1;
		}
		int border = vec.size() / matHeight;
		for (int i = 0; i < matHeight; i++) {
			for (int j = 0; j < matWidth; j++) {
				res[i][j] = vec[j];
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

	void Forward(const int& channelIndex) {
		std::vector < std::vector < double>> matrix;
		int size = blocks[channelIndex]->numOfFilters;
		for (int i = 0; i < size; i++) {
			matrix = photoMatrix;
			//std::cout << "Filter index" << i << std::endl;
			BlokOfConvNPool(channelIndex, i, matrix);
			SetMapOfSigns(channelIndex, i, MatrixIntoVector(matrix));
			//ShowMatrix(matrix);
			matrix.clear();
		}
	}


	void Forward(const int& channelIndex, bool ChannelSum) {
		std::vector < std::vector < double>> matrix;
		std::vector < std::vector < double>> matrixPrev = VectorIntoMatrix(ChannelMatrixSum(channelIndex - 1));
		int size = blocks[channelIndex]->numOfFilters;
		for (int i = 0; i < size; i++) {
			matrix = matrixPrev;
			//std::cout << "Filter index" << i << std::endl;
			BlokOfConvNPool(channelIndex, i, matrix);
			SetMapOfSigns(channelIndex, i, MatrixIntoVector(matrix));
			//ShowMatrix(matrix);
			matrix.clear();
		}
	}

	std::vector<double> ChannelMatrixSum(const int& channelIndex) {
		std::vector<double> temp = GetMapOfSigns(channelIndex, 0);
		for (int i = 1; i < blocks[channelIndex]->numOfFilters; i++) {
			VectorSum(GetMapOfSigns(channelIndex, i), temp);
		}

		return temp;
	}

	void VectorSum(const std::vector<double>& vec, std::vector<double>& res) {
		if (vec.size() == res.size()) {
			for (int i = 0; i < res.size(); i++) {
				res[i] += vec[i];
			}
		}
		else return;
	}

	void Check(const int& channelIndex) {  //rename
		std::vector < std::vector < double>> matrix;
		int size = blocks[channelIndex]->numOfFilters;
		for (int i = 0; i < size; i++) {
			std::cout << "Filter index" << i << std::endl;
			matrix = VectorIntoMatrix(GetMapOfSigns(channelIndex, i));
			//ShowMatrix(matrix);
			matrix.clear();
		}
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
		std::vector<double> vec(matrix[0].size() + 2, 0);
		for (auto& n : matrix) {

			n.insert(n.begin(), 0.0);
			n.push_back(0.0);
		}
		matrix.push_back(vec);
		matrix.insert(matrix.begin(), vec);
	}

	std::vector < std::vector < double>> BilinearInterpolation(std::vector < std::vector < double>>& matrix) {

		std::vector < std::vector < double>> res(imageHeight, std::vector<double>(imageWidth, 0.0));

		double decr—oefX = matrix.size() / imageHeight;
		double decr—oefY = matrix[0].size() / imageWidth;
		for (int x = 0; x < imageHeight; x++) {
			for (int y = 0; y < imageWidth; y++) {
				double scaleX = (x + 0.5) * decr—oefX - 0.5;
				double scaleY = (y + 0.5) * decr—oefY - 0.5;

				int x1 = scaleX;
				int x2 = scaleX + 1;
				int y1 = scaleY;
				int y2 = scaleY + 1;

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

	std::vector<double> FullForward(std::vector < std::vector < double>>& matrix) {
		ChangeMatrixSize(matrix);
		SetImageMatrix(matrix);
		Forward(0);

		for (int i = 1; i < numOfBlocks; i++) {
			Forward(i, true);
		}

		int size = GetFiltersNum(numOfBlocks - 1);
		std::vector<double> res(size, 0);
		for (int i = 0; i < size; i++) {
			res[i] = (GAP(VectorIntoMatrix(GetMapOfSigns(numOfBlocks - 1, i))));
		}

		return res;
	}

	~CNN() {
		for (int i = 0; i < numOfBlocks; i++) {
			delete[] blocks[i];
		}
		delete[] blocks;
	}



	//Activation Functions

	double LeakyReLu(double res) {
		return res > 0 ? res : res * 0.01;
	}

	double DirectiveLeakyReLu(double res) {
		return res > 0 ? res : 0.01;
	}

	double ReLu(double res) {
		return res > 0 ? res : 0;
	}

	double DirectiveReLu(double res) {
		return res > 0 ? 1 : 0;
	}

	//Setter & Getters
	std::vector<double> GetMapOfSigns(int channelIndex, int filterIndex) const {
		return blocks[channelIndex][filterIndex].GetMap();
	}

	int GetMapOfSignsSize(int channelIndex, int filterIndex) const {
		return blocks[channelIndex][filterIndex].GetMapSize();
	}

	void SetMapOfSigns(int channelIndex, int filterIndex, const std::vector<double>& mapOfSigns) const {
		blocks[channelIndex][filterIndex].SetMap(mapOfSigns, mapOfSigns.size());
	}

	void SetImageMatrix(const std::vector<std::vector<double>>& matrix) {
		this->photoMatrix = matrix;
	}

	int GetFiltersNum(int channelIndex) {
		return blocks[channelIndex]->numOfFilters;
	}

};



int main()
{
	auto start = std::chrono::high_resolution_clock::now();
	CNN c;
	std::vector < std::vector < double>> matrix(512, std::vector < double>(512, 127));

	std::vector<double> res = c.FullForward(matrix);

	c.ShowVector(res);

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = end - start;

	std::cout << "Execution time: " << duration.count() << std::endl;
	return 0;

}
