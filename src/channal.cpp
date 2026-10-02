#include "channal.h"

Channal::Channal()
{

}

Channal::~Channal()
{
        delete[] maps;
		delete[] filters;
}

void Channal::SetAmount(int amount) {
		maps = new MapOfSigns[amount];
		for (int i = 0; i < amount; i++) {
			maps[i].mapOfSigns.resize(imageHeight * imageWidth);
		}
		filters = new Filter[amount];
		for (int i = 0; i < amount; i++) {
			filters[i].SetFilter(InitFilterWeight());
		}
		this->amount = amount;
	}

	std::vector<double> Channal::InitFilterWeight() {
		std::vector<double> filter(filterHeight * filterWidth);
		for (auto& n : filter) {
			n = Weight(-0.1, 0.1);
		}
		return filter;
	}

	double Channal::Weight(const double& leftBoard, const double& rightBoard) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(leftBoard, rightBoard);
		return dis(gen);
	}

	std::vector<double> Channal::ChannelSum() {
		std::vector<double> t = maps[0].mapOfSigns;
		for (int i = 1; i < amount - 1; i++) {
			for (int j = 0; j < maps[i].mapOfSigns.size(); j++)
				maps[i].mapOfSigns[j] += maps[i + 1].mapOfSigns[j];
		}
		return t;
	}


	void Channal::DoubleConvMaps(double bias, const std::vector<double>& vec) {
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
					t[mapIdx++] = GeLu(sum);
				}
			}
			mapIdx = 0;
			std::cout << "One conv done\n";
			for (int i = 0; i <= tHeight - 3; i += 3) {
				for (int j = 0; j <= tWidth - 3; j += 3) {
					if (mapIdx >= maps[fIdx].mapOfSigns.size()) break;
					int filter = 0;
					double sum = 0.0;
					int idx = i * tWidth + j;
					for (int kx = 0; kx < 3; kx++) {
						for (int ky = 0; ky < 3; ky++) {
							int newIdx = idx + ((kx * tWidth) + ky);
							if (newIdx >= 0 && newIdx < t.size() && filter < filters[fIdx].filter.size()) {
								sum += t[newIdx] * filters[fIdx].filter[filter++];
							} else {
								
								filter++; 
							}
						}
					}
					sum += bias;
					maps[fIdx].mapOfSigns[mapIdx] = GeLu(sum);
					maps[fIdx].mapOfSignsBeforeActivate[mapIdx] = sum;
					mapIdx++;
				}
			}
			std::cout << "Second conv done\n";
		}
	}

	void Channal::Forward(double bias, const std::vector<double>& vec) {
		DoubleConvMaps(bias, vec);
		Pooling();
		std::cout << "Pooling done\n";
		//CleanZeroFromMap();
	}

	void Channal::RGBForward(double bias, const std::vector<double>& R,
		const std::vector<double>& G, const std::vector<double>& B) {
			std::cout << "Calcul RGB start \n";
		CalculRGBMaps(bias, R, G, B);
		std::cout << "Calcul RGB end \n";
		std::cout << "Pooling RGB start \n";
		Pooling();
		std::cout << "Pooling RGB end \n";
	}

	std::vector<double> Channal::ChannelError() {
		std::vector<double> temp(maps[0].ErrorMapOfSigns.size(), 0.0);

		int width = std::sqrt(maps[0].GetMapOfSigns().size());
		int height = std::sqrt(maps[0].GetMapOfSigns().size());
		for (int i = 0; i < amount; i++) {
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					int idx = x * width + y;
					double error = maps[i].ErrorMapOfSigns[idx];
					for (int kx = 0; kx < 3; kx++) {
						for (int ky = 0; ky < 3; ky++) {
							int idxIn = (y * 3 + kx) * width + (x * 3 + ky);
							int filIdx = kx * 3 + ky;

							temp[idxIn] += error * filters[i].filter[filIdx];
						}
					}
				}
			}
		}
		return temp;
	}


	void Channal::CalculRGBMaps(double bias, const std::vector<double>& R,
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
					maps[q].mapOfSigns[mapIndex++] = GeLu(sum);
				}
			}
		}
	}

	void Channal::ResetMaps() {
		for (int i = 0; i < amount; i++) {
			std::fill(maps[i].mapOfSigns.begin(), maps[i].mapOfSigns.end(), 0.0);
		}
	}

	void Channal::Pooling() {
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

	void Channal::CleanZeroFromMap() {
		for (int i = 0; i < amount; i++) {
			maps[i].mapOfSigns.erase(std::remove(maps[i].mapOfSigns.begin(), maps[i].mapOfSigns.end(), 0.0), maps[i].mapOfSigns.end());
		}
	}

	void Channal::CalculMatrixOfErrors() {
		for (int i = 0; i < amount; i++) {
			maps[i].CalculMatrixOfError(filterHeight, filterHeight);

		}
	}

	void Channal::UpdateFilterWeights() {
		std::vector<double> temp;
		for (int i = 0; i < amount; i++) {
			temp = maps[i].GetMatrixOfError();
			filters[i].UpdateVelocity(temp);
			filters[i].UpdateWeights();
		}
	}


	//New learning variant(remove old and develop this) P.S. not checked and probably incorrect

	void Channal::FilterError(const std::vector<double>& omega, int filterIdx) {
		std::vector<double> tempFilter = filters[filterIdx].GetFilter();
		std::vector<double> error(tempFilter.size(), 0.0);
		for (int i = 0; i < tempFilter.size(); i++) {
			error[i] = omega[i] * tempFilter[i];
		}
		filters[filterIdx].UpdateVelocity(error);
		filters[filterIdx].UpdateWeights();
	}


	bool Channal::CheckIdx(int idx, const std::vector<double>& vec) {
		return (idx >= vec.size() || idx < 0);
	}

	void Channal::InputError(const std::vector<double>& omega, int filterIdx, int maxElementIdx) {
		std::vector<double> temp(omega.size(), 0.0);
		int idx = 0;
		int increaseIdx = -1;
		for (int i = 0; i < omega.size(); i++) {
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

			idx = maxElementIdx + offset + increaseIdx;
			if (CheckIdx(idx, maps[filterIdx].mapOfSigns)) {
				temp[i] = 0.0;
			}
			else {
				temp[i] = omega[i] * maps[filterIdx].mapOfSigns[idx];
			}
			increaseIdx++;
			if (increaseIdx > 1) {
				increaseIdx = -1;
			}
		}

		maps[filterIdx].SetInputError(temp);

	}


	std::vector<double> Channal::BackpropError(const std::vector<double>& dX, int channelIdx) {
		std::vector<double> omega(dX.size(), 0.0);
		for (int i = 0; i < omega.size(); i++) {
			omega[i] = dX[i] * maps[channelIdx].mapOfSigns[i];
		}
		FilterError(omega, channelIdx);
		InputError(omega, channelIdx, maps[channelIdx].GetMaxElIdx());

		return maps[channelIdx].inputError;
	}

	void Channal::ChannalBackprop(const std::vector<double>& E) {
		std::vector<double> temp = E;
		for (int i = 0; i < amount; i++) {
			temp = BackpropError(temp, i);
		}
	}

	//Activate Functions

	double Channal::LeakyReLu(double res) {
		return res > 0.0 ? res : res * 0.01;
	}

	double Channal::DirectiveLeakyReLu(double res) {
		return res > 0.0 ? res : 0.01;
	}

	double Channal::ReLu(double res) {
		return res > 0.0 ? res : 0.0;
	}

	double Channal::DirectiveReLu(double res) {
		return res > 0.0 ? 1 : 0.0;
	}

	double Channal::GeLu(double res) {
		return 0.5 * res * (1 + std::tanh(std::sqrt(2 / std::numbers::pi) * (res + 0.044715 * std::pow(res, 3))));
	}

	double Channal::DirectiveGeLu(double res) {
		double y = std::sqrt(2 / std::numbers::pi) * (res + 0.044715 * std::pow(res, 3));
		return 0.5 * (1 + tanh(y)) + 0.5 * res * (1 - std::pow(tanh(y), 2)) * std::sqrt(2 / std::numbers::pi) * (1 + 0.134145 * std::pow(res, 2));
	}

	//Setter & Getters

	std::vector<double > Channal::GetMap(int index) const {
		return maps[index].GetMapOfSigns();
	}

	void Channal::SetMap(int index, const std::vector<double>& map) {
		maps[index].SetMapOfSigns(map);
	}

	void Channal::SetErrorMap(int index, double error, int errIdx) {
		maps[index].SetErrorMapOfSigns(errIdx, error);
	}

	int Channal::GetNumOfFilters() const {
		return this->amount;
	}

	void Channal::SetMaxElementIndex(int mapIndex, int elIndex) {
		maps[mapIndex].SetMaxElIdx(elIndex);
	}

	int Channal::GetMaxElementIndex(int mapIndex) const {
		return maps[mapIndex].GetMaxElIdx();
	}

	std::vector<double > Channal::GetMatrixOfError(int mapIndex) const {
		return maps[mapIndex].GetMatrixOfError();
	}

	std::vector<double > Channal::GetFilter(int filterIndex) const {
		return filters[filterIndex].GetFilter();
	}