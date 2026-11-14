#include "filters.h"

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
        throw std::string("Invalid size of filter to set");
    }
    else if(HasNanOrInf(filter)){
        throw std::string("Invalid data in filter from paramets");
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

std::vector<double> Filters::Slide(const std::vector<double> &bigMatrix,const std::vector<double> &smallMatrix) {
    if(smallMatrix.size() > bigMatrix.size()){
        throw std::string("Small matrix size is bigger then big matrix size");
    }
    else if(HasNanOrInf(bigMatrix)){
        throw std::string("Big matrix has invalid data in");
    }
    else if(HasNanOrInf(smallMatrix)){
        throw std::string("Small matrix has invalid data in");
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
                            throw std::string("Invalid idx in big matrix\t") + std::to_string((x + kx) * bigMatWidth + (y + ky));
                        }
                        else if((kx * smallMatWidth + ky) >= smallMatrix.size() ){
                            throw std::string("Invalid idx in small matrix\t") + std::to_string(kx * smallMatWidth + ky);
                        }
                        else{
                            sum += bigMatrix[(x + kx) * bigMatWidth + (y + ky)] * smallMatrix[kx * smallMatWidth + ky];
                        }
                    }

                }
                if((newMatX * newMatWidth + newMatY) >= resMatrix.size() ){
                    throw std::string("Invalid idx in res matrix\t") + std::to_string(newMatX * newMatWidth + newMatY);
                }
                else{
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
