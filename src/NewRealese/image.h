#ifndef IMAGE_H
#define IMAGE_H

#pragma once
#include "stb_impl.cpp"
#include <vector>
#include <string>

class Image
{
public:
    Image(std::string imagePath);
    ~Image();

    std::vector<double> LoadImage(std::string imagePath, int color);

    void NormalizeImgColor(std::vector<double> &color);

    std::vector<double> GetRedMatrix() const;
    std::vector<double> GetGreenMatrix() const;
    std::vector<double> GetBlueMatrix() const;

    void ImgPadding(std::vector<double> & matrix);

    void BilinearInterpolation(std::vector < double> & matrix);

private:

std::vector<double> R;
std::vector<double> G;
std::vector<double> B;
int imgHeight = 840;
int imgWidth = 840;

};

#endif