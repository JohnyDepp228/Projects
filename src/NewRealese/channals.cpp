#include "channals.h"

Channals::Channals()
{

}

Channals::~Channals()
{

}
void Channals::SetFiltersParams(int amount,int height,int width) {
    this->filters.resize(amount);
    for(auto &f : filters){
        f.SetSize(height,width);
        f.FillFilter();
    }
}

