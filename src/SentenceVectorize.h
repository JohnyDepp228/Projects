#ifndef SENTENCEVECTORIZE_H
#define SENTENCEVECTORIZE_H



#include <map>
#include <math.h>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <set>
#include <iostream>
#include "DatasetGenerator.h"


struct WordsIDF {
	char* word;
	double IDF;
	int size;
};

class WordsVectorize {
private:
	unsigned int uniqueWordsAmount;
	unsigned int datasetSize;
	std::map<std::string, int> numOfWordsInSentenceInDataset;
	std::map<std::string, double> IDF;
	std::vector<std::string> separateWordsFromSQL;
	WordsIDF *idf;
	std::string path;
	Dataset set;

public:
	WordsVectorize(const std::vector<std::string> &dataset,
		const std::vector<std::string>& methods, const std::vector<std::string>& pages, const std::vector<std::string>& protocol,
		const std::vector<std::string>& agent,
		const std::vector<std::string>& SQL, const std::vector<std::string>& XSS, const std::vector<std::string>& Win) {
		IDF.clear();
		path = "C:/Users/LordMegatron/Desktop/Pupa/idf.txt";
			this->uniqueWordsAmount = set.GetNumOfUniqueWords();
			this->datasetSize = dataset.size();
			separateWordsFromSQL = SentenceIntoSeparateWords(SQL);
			FindsNumOfWordsInDatasetSentences(methods, pages, protocol, agent, separateWordsFromSQL, XSS, Win, dataset);
			calculIDF();
			SaveIDFToFile();
	}

	WordsVectorize() {
		IDF.clear();
		ReadIDFFromFile();
	}

	std::vector<std::string> SentenceIntoSeparateWords(const std::vector<std::string>& SQL);

	void FindsNumOfWordsInDatasetSentences(const std::vector<std::string>& methods, const std::vector<std::string>& pages,
		const std::vector<std::string>& protocol,
		const std::vector<std::string>& agent,
		const std::vector<std::string>& SQL, const std::vector<std::string>& XSS, const std::vector<std::string>& Win,
		std::vector<std::string> dataset);

	void MapIndexInit(std::map<std::string, int> & tempMap,std::vector<std::string> vec);

	void MapIndexInit(std::map<std::string, double> & tempMap, std::vector<std::string> vec);

	void calculIDF();

	~WordsVectorize() {
		for (int i = 0; i < uniqueWordsAmount; i++) {
			delete[] idf[i].word;
		}
		delete[] idf;
	}

	void SaveIDFToFile();

	void ReadIDFFromFile();

	std::vector<double> TFxIDF(std::string word,const unsigned int& ipnutNeuronsAmount);

	std::vector<std::string> SentenceIntoSeparateWords(std::string sentence,int &totalWords);

	std::vector<std::string> SentenceIntoSeparateWords(std::string sentence);

	std::map<std::string, double> TF(std::string sentence);

	void FromStringToChar(std::string sentence,char * arr,int size);
	void FromCharToString(char* arr,std::string sentence,int size);
};
#endif 