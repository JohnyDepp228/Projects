#ifndef CONVNEURNET_H
#define CONVNEURNET_H

#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <numbers> 
#include "stb_image.h"
#include "channal.h"

class ConvNeurNet
{
public:
    ConvNeurNet();

    ~ConvNeurNet();

    std::vector<std::vector<double>> LoadImage(std::string imagePath, int color);

	void UpdateFiltersWeights(int channalIdx);

	void CalculMatrixOfErrorsInChannel(int channalIdx);

	void MarkingErrorsOnMap(const std::vector<double>& fullyconnectedLayerErrors) ;

	void ShowVector(const std::vector<double>& vec);

	void ShowMatrix(const std::vector < std::vector < double>>& matrix);

	double GMP(const std::vector < double>& vec);

	double GMP(const std::vector < double>& vec, int& maxElIndex);

	std::vector < std::vector < double>> VectorIntoMatrix(const std::vector<double>& vec);

	std::vector < double> MatrixIntoVector(const std::vector < std::vector < double>>& matrix);

	void IncreaseMatrixeSize(std::vector < std::vector < double>>& matrix);

	void Padding(std::vector < std::vector < double>>& matrix);

	std::vector < std::vector < double>> BilinearInterpolation(std::vector < std::vector < double>>& matrix);

	void ChangeMatrixSize(std::vector < std::vector < double>>& matrix);

	void ForwardRGB(const std::vector<double>& R, const std::vector<double>& G, const std::vector<double>& B);

	std::vector<double> Forward();

	void NormalizeImage(std::vector<std::vector<double>>& matrix);

	void LearningConvLayers(const std::vector<double>& fullyconnectedLayerErrors);

	//Activation Functions

	double LeakyReLu(double res);

	double DirectiveLeakyReLu(double res);

	double ReLu(double res);

	double DirectiveReLu(double res);

	double GeLu(double res);

	double DirectiveGeLu(double res);

	//Setter & Getters
	std::vector < std::vector < double>> GetPhotoMatrix() const;

	void SetImageMatrix(const std::vector<std::vector<double>>& matrix);

	int GetNumOFBlocks() const;

	std::vector<double> GetMap(int channelInx) const;

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

	Channal* channels;
};

#endif