#ifndef CHANNALS_H
#define CHANNALS_H

#pragma once
#include <vector> 
#include "filters.h"
#include "mapofsigns.h"


class Channals
{
public:
    Channals();
    ~Channals();

    void SetFiltersParams(int amount,int height,int width);



private:
    std::vector<MapOfSigns> maps;
    std::vector<Filters> filters;
};

#endif