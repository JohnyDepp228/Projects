cmake_minimum_required(VERSION 4.1) 
project(Perceptron
	LANGUAGES CXX
	)

add_executable(Perceptron 
    main.cpp 
    perceptron.cpp 
    DatasetGenerator.cpp 
    SaveConfig.cpp 
    SentenceVectorize.cpp
	Test.cpp
)


