cmake_minimum_required(VERSION 4.1) 
project(TicTacToe
	LANGUAGES CXX
	)

add_executable(TicTacToe
    main.cpp 
    perceptron.cpp 
    DatasetGenerator.cpp 
    SaveConfig.cpp 
)


