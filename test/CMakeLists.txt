cmake_minimum_required(VERSION 4.1) 
project(TestPerceptron
	LANGUAGES CXX
	)

add_executable(TestPerceptron 
    main.cpp 
    ../src/server/perceptron.cpp 
    ../src/server/DatasetGenerator.cpp 
    ../src/server/SaveConfig.cpp 
    ../src/server/SentenceVectorize.cpp
    Test.cpp 
)


