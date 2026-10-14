#include "image.h"


enum Color{
    red = 0,
    green = 1,
    blue = 2
};

Image::Image(std::string imagePath)
{
//this->R = LoadImage(imagePath,Color::red);
//this->G = LoadImage(imagePath,Color::green);
//this->B = LoadImage(imagePath,Color::blue);
}

Image::~Image()
{

}

std::vector<double> Image::LoadImage(std::string imagePath, int color) {
		int channels = 0;
		unsigned char* res = stbi_load(imagePath.c_str(), &imgHeightLoaded, &imgWidthLoaded, &channels, 3);

        std::vector<double> img(imgHeightLoaded* imgWidthLoaded);
		if (res == NULL) {
            std::string error = "Error read image\t" + (std::string)stbi_failure_reason();
			throw error;
			exit(1);
		}

		for (int x = 0; x < imgHeightLoaded; x++) {
			for (int y = 0; y < imgWidthLoaded; y++) {
				int pixelIndex = (x * imgWidthLoaded + y) * 3;
                img[pixelIndex] =  (double)res[pixelIndex + color];
			}
		}

		stbi_image_free(res);

		return img;
}

void Image::NormalizeImgColor(std::vector<double> &color) {
    for(auto &pix: color){
        pix /= 255.0;
    }
}

std::vector<double> Image::GetRedMatrix() const { return this->R; }
std::vector<double> Image::GetGreenMatrix() const { return this->G; }
std::vector<double> Image::GetBlueMatrix() const { return this->B; }
void Image::ImgPadding(std::vector<double> &matrix) 
{
	int offsetHeight = (this->imgHeight - this->imgHeightLoaded) /2;
	int offsetWidth = (this->imgWidth - this->imgWidthLoaded) /2;

	std::vector<double> correctSizeMatrix(imgHeight * imgWidth,0.0);

	for (int r = 0; r < this->imgHeight; ++r) {
        
        auto startIt = matrix.begin() + (r * this->imgWidthLoaded);
        auto endIt = startIt + this->imgWidthLoaded;

        int targetRowIdx = r + offsetHeight;
        int targetGlobalIdx = targetRowIdx * this->imgWidth + offsetWidth;

        std::copy(startIt, endIt, correctSizeMatrix.begin() + targetGlobalIdx);
    }

	matrix = correctSizeMatrix;
}
void Image::BilinearInterpolation(std::vector<double> &matrix) 
{


}