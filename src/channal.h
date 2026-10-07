#ifndef CHANNAL_H
#define CHANNAL_H

#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <algorithm>
#include <numbers> 

struct Filter {
	std::vector<double> filter;
	std::vector<double> velocity;
	double LR = 0.01;

	void UpdateVelocity(const std::vector<double>& matrixOfError) {
		for (int i = 0; i < matrixOfError.size(); i++) {
			this->velocity[i] = (0.9 * this->velocity[i]) + (LR * matrixOfError[i]);
		}
	}

	void UpdateWeights() {
		for (int i = 0; i < filter.size(); i++) {
			filter[i] -= velocity[i];
		}
	}

	void SetFilter(const std::vector<double>& filter) {
		this->filter = filter;
		this->velocity.resize(filter.size());
	}

	std::vector<double> GetFilter() const {
		return this->filter;
	}
};

struct MapOfSigns {
	std::vector<double> mapOfSigns;
	std::vector<double> mapOfSignsBeforeActivate;
	std::vector<double> ErrorMapOfSigns;
	std::vector<double> MatrixOfError;
	std::vector<double> inputError;
	double E = 0.0;
	int maxElementIndex = 0;

	void FindE() {
		E = 0.0;
		for (int i = 0; i < ErrorMapOfSigns.size(); i++) {
			E += (mapOfSigns[i] * ErrorMapOfSigns[i]);
		}
	}



	void CalculMatrixOfError(int filterWidth, int filterHeight) {
		FindE();
		MatrixOfError.resize(filterWidth * filterHeight);
		int idx = 0;
		int increaseIdx = -1;
		for (int i = 0; i < MatrixOfError.size(); i++) {
			int offset = 0;
			if (i < 3) {
				offset -= filterHeight;

			}
			else if (i >= 3 && i < 6) {
				offset = 0;

			}
			else if (i > 5) {
				offset = filterHeight;
			}

			idx = maxElementIndex + offset + increaseIdx;
			if (CheckIdx(idx, mapOfSigns)) {
				MatrixOfError[i] = 0;
			}
			else {
				MatrixOfError[i] = E * mapOfSigns[idx];
			}
			increaseIdx++;
			if (increaseIdx > 1) {
				increaseIdx = -1;
			}
		}
	}

	bool CheckIdx(int idx, const std::vector<double>& vec) {
		return (idx >= vec.size() || idx < 0);
	}

	void SetMapOfSigns(const std::vector<double>& mapOfSigns) {
		this->mapOfSigns.resize(mapOfSigns.size());
		this->ErrorMapOfSigns.resize(mapOfSigns.size(), 0.0);
		this->mapOfSigns = mapOfSigns;

	}

	void SetMapBeforeActivate(const std::vector<double>& mapOfSignsBeforeActivate) {
		this->mapOfSignsBeforeActivate = mapOfSignsBeforeActivate;
	}

	void SetErrorMapOfSigns(int idx, double error) {
		this->ErrorMapOfSigns[idx] = error;
	}

	std::vector<double> GetMapOfSigns() const {
		return this->mapOfSigns;
	}

	void SetMaxElIdx(int maxElementIndex) {
		this->maxElementIndex = maxElementIndex;
	}

	void SetInputError(const std::vector<double>& inputError) {
		this->inputError = inputError;
	}

	int GetMaxElIdx() const {
		return this->maxElementIndex;
	}

	std::vector<double> GetMatrixOfError() const {
		return this->MatrixOfError;
	}

};

class Channal
{
public:
    Channal();

    ~Channal();

    void SetAmount(int amount);

	std::vector<double> InitFilterWeight();

	double Weight(const double& leftBoard, const double& rightBoard);

	std::vector<double> ChannelSum();


	void DoubleConvMaps(double bias, const std::vector<double>& vec);

	void Forward(double bias, const std::vector<double>& vec);

	void RGBForward(double bias, const std::vector<double>& R,
		const std::vector<double>& G, const std::vector<double>& B);

	std::vector<double> ChannelError();


	void CalculRGBMaps(double bias, const std::vector<double>& R,
		const std::vector<double>& G, const std::vector<double>& B);

	void ResetMaps();

	void Pooling();

	void CleanZeroFromMap();

	void CalculMatrixOfErrors();

	void UpdateFilterWeights();


	//New learning variant(remove old and develop this) P.S. not checked and probably incorrect

	void FilterError(const std::vector<double>& omega, int filterIdx);


	bool CheckIdx(int idx, const std::vector<double>& vec);

	void InputError(const std::vector<double>& omega, int filterIdx, int maxElementIdx);


	std::vector<double> BackpropError(const std::vector<double>& dX, int channelIdx);

	void ChannalBackprop(const std::vector<double>& E);

	//Activate Functions

	double LeakyReLu(double res);

	double DirectiveLeakyReLu(double res);

	double ReLu(double res);

	double DirectiveReLu(double res);

	double GeLu(double res);

	double DirectiveGeLu(double res);

	//Setter & Getters

	std::vector<double > GetMap(int index) const;

	void SetMap(int index, const std::vector<double>& map);

	void SetErrorMap(int index, double error, int errIdx);

	int GetNumOfFilters() const;

	void SetMaxElementIndex(int mapIndex, int elIndex);

	int GetMaxElementIndex(int mapIndex) const;

	std::vector<double > GetMatrixOfError(int mapIndex) const;

	std::vector<double > GetFilter(int filterIndex) const;

private:
    MapOfSigns* maps = nullptr;
	Filter* filters = nullptr;
	int amount = 0;
	int filterHeight = 3;
	int filterWidth = 3;
	int imageHeight = 840;
	int imageWidth = 840;
	int mapHeight = 0;
	int mapWidth = 0;
};

#endif