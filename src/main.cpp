#include "classifier.h"
#include "convneurnet.h"


int Predict(std::string path) {
	int datasetSize = 1000;
	std::vector<double> correcrIndexOfNeurons(1000, 0.0);
	correcrIndexOfNeurons[98] = 1.0;
	ConvNeurNet c;
	Classifier cl;
	std::vector < std::vector < double>> matrixR = c.LoadImage(path, 0);

	if(matrixR.size() > 0){
		std::cout << "Success in red" << std::endl;
	}
	std::vector < std::vector < double>> matrixG = c.LoadImage(path, 1);
	if(matrixG.size() > 0){
		std::cout << "Success in green" << std::endl;
	}
	std::vector < std::vector < double>> matrixB = c.LoadImage(path, 2);
	if(matrixB.size() > 0){
		std::cout << "Success in blue" << std::endl;
	}
	c.NormalizeImage(matrixR);
	c.NormalizeImage(matrixG);
	c.NormalizeImage(matrixB);
	c.ChangeMatrixSize(matrixR);
	c.ChangeMatrixSize(matrixG);
	c.ChangeMatrixSize(matrixB);

	std::vector<double> R = c.MatrixIntoVector(matrixR);
	std::vector<double> G = c.MatrixIntoVector(matrixG);
	std::vector<double> B = c.MatrixIntoVector(matrixB);


	std::cout << "Forward of RGB start" << std::endl;

	c.ForwardRGB(R, G, B);

	std::cout << "Forward of RGB success" << std::endl;

	std::vector<double> res = c.Forward();

	cl.SetFullyConnectedLayer(res);
	cl.Classification();
	std::cout << "Learning..." << std::endl;
	cl.LearningClassifier(98);
	c.LearningConvLayers(cl.GetInputErrors());
	return cl.FindCorrectOutNeuro();
}


int main()
{
	auto start = std::chrono::high_resolution_clock::now();

	std::string path = "C:/Users/Boss/Desktop/с флешки/2024_12_13 FOTO/13_12_0954.jpg";
	std::string path2 = "C:/Users/LordMegatron/Desktop/2.jpg";

	std::cout << Predict(path2) << std::endl;

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = end - start;

	std::cout << "Execution time: " << duration.count() << std::endl;
	//system("pause");

	return 0;

}
