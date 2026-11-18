#include "filters.h"
#include <algorithm>
#include <cmath>

Filters::Filters()
{

}

Filters::~Filters()
{

}
void Filters::SetHeight(unsigned int height) {
    this->height = height;
}

void Filters::SetWidth(unsigned int width) {
    this->width = width;
}

void Filters::SetFilter(const std::vector<double> &filter) {
    if(filter.size() < 1){
        throw std::string("Invalid size of filter to set(filters learning)");
    }
    else if(HasNanOrInf(filter)){
        throw std::string("Invalid data in filter from paramets(filters learning)");
    }
    else{
        this->filter = filter;
    }
}

void Filters::SetStride(int stride) {
    this->stride = stride;
}

unsigned int Filters::GetHeight() const {
    return this->height;
}

unsigned int Filters::GetWidth() const {
    return this->width;
}

std::vector<double> Filters::GetFilter() const {
    return this->filter;
}

int Filters::GetStride() const {
    return this->stride;
}

void Filters::FillFilter() {
    this->filter.resize(this->width * this->height);
    this->padding = this->height - this->stride;
    double bound = std::sqrt(2.0 / (this->width * this->height)); 
    for(auto &f : this->filter){
        f = RandValue(-bound, bound);
    }
}

double Filters::RandValue(double leftBorder,double rightBorder) {
    std::random_device rd;
    std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(leftBorder, rightBorder);
	return dis(gen);
}

bool Filters::HasNanOrInf(const std::vector<double> &vec) {
    	for (double n : vec) if (std::isnan(n) || std::isinf(n)) return true;
    	return false;
}

std::vector<double> Filters::Slide(const std::vector<double> &bigMatrix,const std::vector<double> &smallMatrix,bool addBias) {
    if(smallMatrix.size() > bigMatrix.size()){
        throw std::string("Small matrix size is bigger then big matrix size(filters learning)");
    }
    else if(HasNanOrInf(bigMatrix)){
        throw std::string("Big matrix has invalid data in(filters learning)");
    }
    else if(HasNanOrInf(smallMatrix)){
        throw std::string("Small matrix has invalid data in(filters learning)");
    }
    else{
        int bigMatHeight;
        int bigMatWidth = bigMatHeight = std::sqrt(bigMatrix.size());
        int smallMatHeight;
        int smallMatWidth = smallMatHeight = std::sqrt(smallMatrix.size());
        int newMatX = 0;
        int newMatHeight = ((bigMatHeight - smallMatHeight) / this->stride) + 1;
        int newMatWidth =  ((bigMatWidth - smallMatWidth) / this->stride) + 1;
        std::vector<double> resMatrix(newMatHeight * newMatWidth,0.0);
        for(int x = 0; x <= bigMatHeight - smallMatHeight; x += this->stride){
            int newMatY = 0;
            for(int y = 0; y <= bigMatWidth - smallMatWidth; y += this->stride){

                double sum = 0.0;
                
                for(int kx = 0; kx < smallMatHeight; kx++){

                    for(int ky = 0; ky < smallMatWidth; ky++){
                        if(((x + kx)* bigMatWidth + (y + ky)) >= bigMatrix.size() ){
                            throw std::string("Invalid idx in big matrix(filters learning)\t") + std::to_string((x + kx) * bigMatWidth + (y + ky));
                        }
                        else if((kx * smallMatWidth + ky) >= smallMatrix.size() ){
                            throw std::string("Invalid idx in small matrix(filters learning)\t") + std::to_string(kx * smallMatWidth + ky);
                        }
                        else{
                            sum += bigMatrix[(x + kx) * bigMatWidth + (y + ky)] * smallMatrix[kx * smallMatWidth + ky];
                        }
                    }

                }
                if((newMatX * newMatWidth + newMatY) >= resMatrix.size() ){
                    throw std::string("Invalid idx in res matrix(filters learning)\t") + std::to_string(newMatX * newMatWidth + newMatY);
                }
                else{
                    if(addBias) sum += this->bias;
                    resMatrix[newMatX * newMatWidth + newMatY] = sum;
                   
                }

                newMatY++;
            }
            newMatX++;
        }
        return resMatrix;
    }

    return{};
}

void Filters::LearnFilters(const std::vector<double> &mapOfSigns,const std::vector<double> &errorMap) {
    if(HasNanOrInf(mapOfSigns)){
        throw std::string("mapOfSigns matrix has invalid data in(filters learning)");
    }
    if(HasNanOrInf(errorMap)){
        throw std::string("errorMap matrix has invalid data in(filters learning)");
    }
    std::vector<double> filtersError = Slide(mapOfSigns,errorMap,0);
    if(HasNanOrInf(filtersError)){
        throw std::string("filtersError matrix has invalid data in");
    }
    filter = (*this) - filtersError;
}

std::vector<double> Filters::FindMapForPrevLayer(const std::vector<double> &errorMap) {
    if(HasNanOrInf(errorMap)){
        throw std::string("errorMap matrix has invalid data in(filters learning)");
    }
    std::vector<double> flippedFilter(filter.size());
    std::reverse_copy(filter.begin(), filter.end(),flippedFilter.begin());
    //std::vector<double> paddedErrorMap = Padding(errorMap);
    std::vector<double> res =  Slide(errorMap,flippedFilter,0);
    if(HasNanOrInf(res)){
        throw std::string("Invalid data in map of error for prev layer(filters learn)");
    }
    else{
        return res;
    }
}

std::vector<double> Filters::operator-(const std::vector<double> &filtersError) {
    if(filtersError.size() != filter.size()){
        throw std::string("Invalid size of filtersError vector(filters learning)");
    }
    else{
        std::vector<double> res(filter.size());
        for(int i =0;i < filter.size();i++){
            res[i] = filter[i] - (LR * filtersError[i]);
        }
        return res;
    }
}

std::vector<double> Filters::Padding(const std::vector<double> &errorMap) {
    int errorMapSizeOrg = std::sqrt(errorMap.size());
    int errorMapSize = 0;
    errorMapSize += 2 * this->padding;
	std::vector<double> correctSizeMatrix(errorMapSize * errorMapSize,0.0);

	for (int r = 0; r < errorMapSizeOrg; ++r) {
        
        auto startIt = errorMap.begin() + (r * errorMapSizeOrg);
        auto endIt = startIt + errorMapSizeOrg;

        int targetRowIdx = r + this->padding;
        int targetGlobalIdx = targetRowIdx * errorMapSize + this->padding;

        std::copy(startIt, endIt, correctSizeMatrix.begin() + targetGlobalIdx);
    }

	return correctSizeMatrix;
}
