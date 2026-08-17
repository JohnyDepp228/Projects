#include "Test.h"

int main() {

	Tests test;
	test.Test(100,80,1, "../../../config/weights.txt", "../../../config/idf.txt");
	return 0;
}