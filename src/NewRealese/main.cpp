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
std::cout << "Mat size before pooling:\t" << mat.size() <<std::endl;
newMat = m.Pooling(mat);
std::cout << "Mat size after pooling:\t" << newMat.size() <<std::endl;
std::vector<double> w = newMat;
std::vector<double> c;
c = m.ReversePooling(w);
std::cout << "Mat size after reverse pooling:\t" << c.size() <<std::endl;


auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = end - start;
std::cout << "Execution time: " << duration.count() << std::endl;

return 0;
}