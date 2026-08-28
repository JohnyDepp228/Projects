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

struct Channal {
	std::vector<double> mapOfSigns;

	std::vector<double> filter;

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
	int poolingWindowWeight = 2;


	int numOfBlocks = 6;
	int numOfFiltersInBlock = 16;
	Channal** blocks;


public:
	CNN() {
		blocks = new Channal * [6];
		for (int i = 0; i < numOfBlocks; i++) {
			blocks[i] = new Channal[numOfFiltersInBlock];
			for (int j = 0; j < numOfFiltersInBlock; j++) {
				blocks[i][j].SetFilter(InitFilterWeight(numOfFiltersInBlock));
			}
			numOfFiltersInBlock *= 2;
		}

	}

	std::vector<double> InitFilterWeight(int filterSize) {
		std::vector<double> filter(filterSize, 0);
		for (auto& n : filter) {
			n = Weight(-0.5, 0.5);
		}
		return filter;
	}

	double Weight(double leftBoard, double rightBoard) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(leftBoard, rightBoard);
		return dis(gen);
	}


	std::vector<double> GetSmallerMatrixFromMatrix(int startI, int startJ, std::vector<std::vector<double>> matrix, int smallMatrixHeight, int smallMatrixWeight) {
		double res = 0.0;
		std::vector<double> temp;
		for (int i = startI; i < startI + smallMatrixHeight; i++) {
			for (int j = startJ; j < startJ + smallMatrixWeight; j++) {
				if (i < matrix.size() && j < matrix[0].size()) {
					temp.push_back(matrix[i][j]);
				}
			}
		}
		return temp;
	}

	double ÑonvolutionOfOneMatrix(std::vector<double> MatrixFromPhoto, int channalIndex, int filterIndex) {
		double res = 0.0;
		std::vector<double> filter = blocks[channalIndex][filterIndex].GetFilter();
		for (int i = 0; i < MatrixFromPhoto.size(); i++) {
			res += MatrixFromPhoto[i] * filter[i];
		}
		res += bias;

		return res;
	}

	double MaxPooling(const std::vector<double>& smallMapOfSigns) {
		if (smallMapOfSigns.empty()) {
			return 0.0;
		}
		else {
			return *max_element(smallMapOfSigns.begin(), smallMapOfSigns.end());
		}
	}

	double MinPooling(std::vector<double> smallMapOfSigns) {
		if (smallMapOfSigns.empty()) {
			return 0.0;
		}
		else {
			return *min_element(smallMapOfSigns.begin(), smallMapOfSigns.end());
		}
	}

	std::vector < std::vector < double>> FullConvolution(const std::vector < std::vector < double>>& matrix, int biggerMatHeight, int biggerMatWiedth,
		int smallerMatHeight, int smallerMatWeight, int& newHeight, int& newWeight, int channalIndex, int filterIndex) {
		newHeight = (biggerMatHeight - smallerMatHeight) + 1;
		newWeight = (biggerMatWiedth - smallerMatWeight) + 1;
		std::vector < std::vector < double>> mapOfSigns((biggerMatHeight - smallerMatHeight) + 1, std::vector < double>((biggerMatWiedth - smallerMatWeight) + 1, 0));
		for (int i = 0; i <= biggerMatHeight - smallerMatHeight; i++) {
			for (int j = 0; j <= biggerMatWiedth - smallerMatWeight; j++) {
				mapOfSigns[i][j] = ÑonvolutionOfOneMatrix(GetSmallerMatrixFromMatrix(i, j, matrix, smallerMatHeight, smallerMatWeight), channalIndex, filterIndex);
			}
		}

		return mapOfSigns;
	}

	void BlokOfConvNPool(int channalIndex, int filterIndex, std::vector < std::vector < double>>& newFilter) {
		int newHeight = 0;
		int newWeight = 0;
		newFilter = FullConvolution(newFilter, newFilter.size(), newFilter[0].size(), filterHeight, filterWeight, newHeight, newWeight, channalIndex, filterIndex);
		newFilter = FullConvolution(newFilter, newHeight, newWeight, filterHeight, filterWeight, newHeight, newWeight, channalIndex, filterIndex);
		newFilter = Pooling(newFilter, poolingWindowHeight, poolingWindowWeight, newHeight, newWeight);
	}

	std::vector < std::vector < double>> GetPhotoMatrix() const {
		return photoMatrix;
	}

	std::vector < std::vector < double>> Pooling(std::vector < std::vector < double>> mapOfSigns, int smallerMatHeight, int smallerMatWeight, int newH, int newW) {
		std::vector < std::vector < double>> smallerMapOfSigns(newH - smallerMatHeight + 1, std::vector < double>(newW - smallerMatWeight + 1, 0));
		for (int i = 0; i <= newH - smallerMatHeight; i++) {
			for (int j = 0; j <= newW - smallerMatWeight; j++) {
				smallerMapOfSigns[i][j] = MaxPooling(GetSmallerMatrixFromMatrix(i, j, mapOfSigns, 2, 2));
			}
		}

		return smallerMapOfSigns;
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
		int matWeight = std::sqrt((double)vec.size());
		std::vector < std::vector < double>> res(matHeight, std::vector<double>(matWeight, 0));
		int border = vec.size() / matHeight;
		for (int i = 0; i < matHeight; i++) {
			for (int j = 0; j < matWeight; j++) {
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

	void Forward(int channalIndex) {
		std::vector < std::vector < double>> matrix;
		std::vector<double> filter = blocks[channalIndex]->GetFilter();
		for (int i = 0; i < filter.size(); i++) {
			matrix = photoMatrix;
			std::cout << "Filter index" << i << std::endl;
			BlokOfConvNPool(channalIndex, i, matrix);
			BlokOfConvNPool(channalIndex, i, matrix);
			SetMapOfSigns(channalIndex, i, MatrixIntoVector(matrix));
			ShowMatrix(matrix);
			matrix.clear();
		}
	}


	void Check(int channalIndex) {  //rename
		std::vector < std::vector < double>> matrix;
		std::vector<double> filter = blocks[channalIndex]->GetFilter();
		for (int i = 0; i < filter.size(); i++) {
			std::cout << "Filter index" << i << std::endl;
			matrix = VectorIntoMatrix(GetMapOfSigns(channalIndex, i));
			ShowMatrix(matrix);
			matrix.clear();
		}
	}

	~CNN() {
		for (int i = 0; i < numOfBlocks; i++) {
			delete[] blocks[i];
		}
		delete[] blocks;
	}



	//Setter & Getters
	std::vector<double> GetMapOfSigns(int channalIndex, int filterIndex) const {
		return blocks[channalIndex][filterIndex].GetMap();
	}

	int GetMapOfSignsSize(int channalIndex, int filterIndex) const {
		return blocks[channalIndex][filterIndex].GetMapSize();
	}

	void SetMapOfSigns(int channalIndex, int filterIndex, const std::vector<double>& mapOfSigns) const {
		blocks[channalIndex][filterIndex].SetMap(mapOfSigns, mapOfSigns.size());
	}

	void SetImageMatrix(const std::vector<std::vector<double>>& matrix) {
		this->photoMatrix = matrix;
	}

};



int main()
{
	auto start = std::chrono::high_resolution_clock::now();
	CNN c;

	std::vector < std::vector < double>> matrix(16, std::vector < double>(16, 0.01));
	InitMatrix(matrix);
	c.SetImageMatrix(matrix);

	c.Forward(0);

	c.Check(0);

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = end - start;

	std::cout << "Âðåìÿ âûïîëíåíèÿ: " << duration.count() << std::endl;
	return 0;
}
