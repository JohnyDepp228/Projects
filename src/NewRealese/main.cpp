#include "filters.h"
#include <random>
#include <chrono>
#include "mapofsigns.h"


double Rand(){
    std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(0.0, 10.0);
		return dis(gen);
}

int main(){
    
Filters f;
MapOfSigns m;
//f.SetSize(3,3);
std::vector<double> mat(705600);
std::vector<double> newMat;
std::cout << std::endl << std::endl << std::endl;
auto start = std::chrono::high_resolution_clock::now();
newMat = f.Slide(mat, 1);
mat = f.Slide(newMat,1);
newMat = m.Pooling(mat);

mat = f.Slide(newMat, 1);
newMat = f.Slide(mat,1);
mat = m.Pooling(newMat);

newMat = f.Slide(mat, 1);
mat = f.Slide(newMat,1);
newMat = m.Pooling(mat);

mat = f.Slide(newMat, 1);
newMat = f.Slide(mat,1);
mat = m.Pooling(newMat);

newMat = f.Slide(mat, 1);
mat = f.Slide(newMat,1);
newMat = m.Pooling(mat);

mat = f.Slide(newMat, 1);
newMat = f.Slide(mat,1);
mat = m.Pooling(newMat);

auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = end - start;
std::cout << "Execution time: " << duration.count() << std::endl;
std::cout << mat.size();
std::vector<int> t = m.PoolingMaxIdx();
for(auto n: t){
    std::cout << n << "\t";
}
}