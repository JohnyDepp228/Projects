#include "filters.h"

Filters::Filters()
{
    SetSize(3,3);
    std::cout << GetElement(0,1);
}

Filters::~Filters()
{
    
}

 void Filters::SetSize(int height,int width){
    this->width = width;
    this->height = height;
    this->filter.resize(this->width * this->height,0.0);
 }

//  double& Filters::operator()(int x,int y)
//     {
//         return this->filter[x * this->width + y];
//     }

double Filters::GetElement(int x,int y) const
    {
        return this->filter[(x * this->width) + y]; 
    }