#include <iostream>
#include <vector>


class CNN {
private:
	std::vector<std::vector<double>> filters;
	std::vector<std::vector<double>> photoMatrix;
	double bias;


public:
	std::vector<double> GetSmallerMatrixFromMatrix(int start, std::vector<std::vector<double>> matrix) {
		int mStart = start;
		double res = 0.0;
		std::vector<double> temp;
		for (int i = 0; i < start + 2; i++) {
			for (int j = mStart; j < mStart + 2; j++) {
				temp.push_back(matrix[i][j]);
			}
		}
		return temp;
	}

	double ÑonvolutionOfOneMatrix(std::vector<double> MatrixFromPhoto, int filterIndex) {
		double res = 0.0;
		for (int i = 0; i < MatrixFromPhoto.size(); i++) {
			double res = MatrixFromPhoto[i] * filters[filterIndex][i];
		}
		res += bias;

		return res;
	}

	double MaxPooling(std::vector<double> smallMapOfSigns) {
		return *max_element(smallMapOfSigns.begin(), smallMapOfSigns.end());
	}

	double MaxPooling(std::vector<double> smallMapOfSigns) {
		return *min_element(smallMapOfSigns.begin(), smallMapOfSigns.end());
	}

	std::vector < std::vector < double>> FullConvolution(int height, int width) {
		std::vector < std::vector < double>> mapOfSigns(height, std::vector < double>(width, 0));
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				mapOfSigns[i][j] = ÑonvolutionOfOneMatrix(GetSmallerMatrixFromMatrix(i, photoMatrix), i);
			}
		}

		return mapOfSigns;
	}

	std::vector < std::vector < double>> Pooling(std::vector < std::vector < double>> mapOfSigns, int height, int width) {
		std::vector < std::vector < double>> smallerMapOfSigns(height, std::vector < double>(width, 0));
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				smallerMapOfSigns[i][j] = MaxPooling(GetSmallerMatrixFromMatrix(i, mapOfSigns));
			}
		}

		return smallerMapOfSigns;
	}
};

int main()
{

}
