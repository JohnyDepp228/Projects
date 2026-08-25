#include <iostream>
#include <vector>


void InitMatrix(std::vector < std::vector < double>>& matrix) {
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			matrix[i][j] = j;
		}
	}
}

class CNN {
private:
	std::vector<std::vector<double>> filters;
	std::vector<std::vector<double>> photoMatrix;
	double bias = 1.0;

	int matSizeAfterConvH = 0;
	int matSizeAfterConvW = 0;


public:
	void SetImageMatrix(const std::vector<std::vector<double>>& matrix) {
		this->photoMatrix = matrix;
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

	double ÑonvolutionOfOneMatrix(std::vector<double> MatrixFromPhoto, int filterIndex) {
		double res = 0.0;
		for (int i = 0; i < MatrixFromPhoto.size(); i++) {
			//double res = MatrixFromPhoto[i] * filters[filterIndex][i];
			res += MatrixFromPhoto[i];
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

	std::vector < std::vector < double>> FullConvolution(int biggerMatHeight, int biggerMatWidth, int smallerMatHeight, int smallerMatWeight) {
		this->matSizeAfterConvH = (biggerMatHeight - smallerMatHeight) + 1;
		this->matSizeAfterConvW = (biggerMatWidth - smallerMatWeight) + 1;
		std::vector < std::vector < double>> mapOfSigns((biggerMatHeight - smallerMatHeight) + 1, std::vector < double>((biggerMatWidth - smallerMatWeight) + 1, 0));
		for (int i = 0; i <= biggerMatHeight - smallerMatHeight; i++) {
			for (int j = 0; j <= biggerMatWidth - smallerMatWeight; j++) {
				mapOfSigns[i][j] = ÑonvolutionOfOneMatrix(GetSmallerMatrixFromMatrix(i, j, photoMatrix, smallerMatHeight, smallerMatWeight), i);
			}
		}

		return mapOfSigns;
	}

	std::vector < std::vector < double>> Pooling(std::vector < std::vector < double>> mapOfSigns, int smallerMatHeight, int smallerMatWeight) {
		std::vector < std::vector < double>> smallerMapOfSigns(matSizeAfterConvH - smallerMatHeight + 1, std::vector < double>(matSizeAfterConvW - smallerMatWeight + 1, 0));
		for (int i = 0; i <= matSizeAfterConvH - smallerMatHeight; i++) {
			for (int j = 0; j <= matSizeAfterConvW - smallerMatWeight; j++) {
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

};



int main()
{
	CNN c;
	std::vector<double> vec;
	std::vector < std::vector < double>> matrix(6, std::vector < double>(6, 0));
	InitMatrix(matrix);
	c.SetImageMatrix(matrix);
	c.ShowMatrix(matrix);
	std::cout << std::endl;
	c.ShowMatrix(c.Pooling(c.FullConvolution(6, 6, 3, 3), 2, 2));
}
