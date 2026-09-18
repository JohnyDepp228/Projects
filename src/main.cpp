#include "perceptron.h"
#include <Windows.h>
#include <iomanip>


enum Turn {
	zero = 0,
	cross = 1,
	empty = -1
};


void ShowMap(const std::vector<double>& map) {
	std::cout << "+---------+---------+---------+" << std::endl;
		for (int i = 0; i < map.size(); i++) {
			char ch = ' ';
			if (map[i] == zero) {
				ch = 'O';
			}
			else if (map[i] == cross) {
				ch = 'X';
			}
			std::cout << "| "
				<< std::setw(7) << std::fixed << std::setprecision(3)
				<< ch
				<< " ";

			if (i % 3 == 2) {
				std::cout << "|" << std::endl;
				std::cout << "+---------+---------+---------+" << std::endl;
			}
		}
}

void MakeStep(std::vector<double>& map,int turn) {
	ShowMap(map);
	std::cout << std::endl << "Enter your step: ";
	int x = 0;
	std::cin >> x;
	x--;
	if (x < 0 || x > map.size()) {
		std::cout << "Invalid step" << std::endl;
	}
	else if(map[x] != Turn::empty) {
		std::cout << "Invalid step" << std::endl;
	}
	else {
		map[x] = turn;
	}

}

void MakePerceptronStep(std::vector<double>& map, int index) {
	map[index] = Turn::zero;

}

int GetWinner(const std::vector<double>& map) {
	int lines[8][3] =
	{
		{0, 1, 2},
		{3, 4, 5},
		{6, 7, 8},

		{0, 3, 6},
		{1, 4, 7},
		{2, 5, 8},

		{0, 4, 8},
		{2, 4, 6}
	};

	for (int i = 0; i < 8; i++)
	{
		int a = lines[i][0];
		int b = lines[i][1];
		int c = lines[i][2];

		if (map[a] != -1 &&
			map[a] == map[b] &&
			map[b] == map[c])
		{
			return (int)map[a];
		}
	}

	return -1;
}

bool IsGameOver(const std::vector<double>& map) {
	if (GetWinner(map) != -1)
		return true;

	for (int i = 0; i < 9; i++)
	{
		if (map[i] == -1)
			return false;
	}

	return true;
}

int main() {
	
	Perceptron p(9, 64, 9);
	std::vector<double> map(9, -1.0);
	std::cout << "You playing for X" << std::endl;
	char ch = ' ';
	for (int i = 0; i < map.size(); i++) {
		MakeStep(map, Turn::cross);
		p.FullProcess(map);
		int answer = p.GetAnswer();
		MakePerceptronStep(map, answer);

		if (IsGameOver(map)) {
			if (GetWinner(map) == zero) {
				ch = 'O';
			}
			else if (GetWinner(map) == cross) {
				ch = 'X';
			}
			std::cout <<"Winner is " << ch << std::endl;
			break;
		}
	}

	return 0;
}