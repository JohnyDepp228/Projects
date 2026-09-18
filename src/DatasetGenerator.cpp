#include "DatasetGenerator.h"


int GenerateDataset::GetWinner(const std::vector<double>& map) {
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

bool GenerateDataset::IsGameOver(const std::vector<double>& map) {
    if (GetWinner(map) != -1)
        return true;

    for (int i = 0; i < 9; i++)
    {
        if (map[i] == -1)
            return false;
    }

    return true;
}

int GenerateDataset::GetRandomMove(const std::vector<double>& map) {
    std::vector<int> freeCells;

    for (int i = 0; i < 9; i++)
    {
        if (map[i] == -1)
            freeCells.push_back(i);
    }

    if (freeCells.empty())
        return -1;

    return freeCells[rand() % freeCells.size()];
}

int GenerateDataset::Minimax(std::vector<double>& map, bool oTurn) {
    int winner = GetWinner(map);

    if (winner == 0)
        return 10;

    if (winner == 1)
        return -10;

    bool full = true;

    for (int i = 0; i < 9; i++)
    {
        if (map[i] == -1)
        {
            full = false;
            break;
        }
    }

    if (full)
        return 0;

    if (oTurn)
    {
        int best = -1000;

        for (int i = 0; i < 9; i++)
        {
            if (map[i] == -1)
            {
                map[i] = 0;

                int score = Minimax(map, false);

                map[i] = -1;

                if (score > best)
                    best = score;
            }
        }

        return best;
    }
    else
    {
        int best = 1000;

        for (int i = 0; i < 9; i++)
        {
            if (map[i] == -1)
            {
                map[i] = 1;

                int score = Minimax(map, true);

                map[i] = -1;

                if (score < best)
                    best = score;
            }
        }

        return best;
    }
}

int GenerateDataset::GetBestMove(std::vector<double>& map) {
    int bestMove = -1;
    int bestScore = -1000;

    for (int i = 0; i < 9; i++)
    {
        if (map[i] == -1)
        {
            map[i] = 0;

            int score = Minimax(map, false);

            map[i] = -1;

            if (score > bestScore)
            {
                bestScore = score;
                bestMove = i;
            }
        }
    }

    return bestMove;
}

void GenerateDataset::GenerateGame(int& datasetIndex) {
    std::vector<double> map(9, -1);

    bool xTurn = true;

    while (!IsGameOver(map))
    {
        if (xTurn)
        {
            int move = GetRandomMove(map);

            if (move == -1)
                break;

            map[move] = 1;

            xTurn = false;
        }
        else
        {
            if (datasetIndex >= datasetSize)
                break;
            std::vector<double> currentMap = map;

            int correctAnswer = GetBestMove(map);

            if (correctAnswer == -1)
                break;

            dataset[datasetIndex].SetMap(currentMap);
            dataset[datasetIndex].SetAnswer(correctAnswer);

            datasetIndex++;

            map[correctAnswer] = 0;

            xTurn = true;
        }
    }
}

void GenerateDataset::CreateDataset() {
    int datasetIndex = 0;
    std::cout << "Dataset gen start " << std::endl;
    while (datasetIndex < datasetSize)
    {   
        std::cout << "Dataset index " << datasetIndex << std::endl;
        GenerateDataset::GenerateGame(datasetIndex);
    }
    std::cout << "Dataset gen done " << std::endl;
}