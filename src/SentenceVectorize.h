#ifndef SENTENCEVECTORIZE_H
#define SENTENCEVECTORIZE_H



#include <map>
#include <math.h>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>


struct WordsIDF {
	std::string word;
	double IDF;
};

class WordsVectorize {
private:
	unsigned int uniqueWordsAmount;
	unsigned int datasetSize;
	std::map<std::string, int> numOfWordsInSentenceInDataset;
	std::map<std::string, int> IDF;
	std::vector<std::string> separateWordsFromSQL;
	WordsIDF *idf;
	std::string path;

public:
	WordsVectorize(const unsigned int& uniqueWordsAmount,std::vector<std::string> dataset,
		const std::vector<std::string>& methods, const std::vector<std::string>& pages, const std::vector<std::string>& protocol,
		const std::vector<std::string>& agent,
		const std::vector<std::string>& SQL, const std::vector<std::string>& XSS, const std::vector<std::string>& Win) {
		this->uniqueWordsAmount = uniqueWordsAmount;
		this->datasetSize = dataset.size();
		separateWordsFromSQL = SentenceIntoSeparateWords(SQL);
		idf = new WordsIDF[uniqueWordsAmount];
	}

	std::vector<std::string> SentenceIntoSeparateWords(const std::vector<std::string>& SQL);

	void FindsNumOfWordsInDatasetSentences(const std::vector<std::string>& methods, const std::vector<std::string>& pages,
		const std::vector<std::string>& protocol,
		const std::vector<std::string>& agent,
		const std::vector<std::string>& SQL, const std::vector<std::string>& XSS, const std::vector<std::string>& Win,
		std::vector<std::string> dataset);

	void MapIndexInit(std::map<std::string, int> numOfWordsInSentenceInDataset,std::vector<std::string> vec);

	void calculIDF();

	~WordsVectorize() {
		delete[] idf;
	}

	void SaveIDFToFile();

	void ReadIDFFromFile();

	std::vector<double> TFxIDF(std::string word,const unsigned int& ipnutNeuronsAmount);

	std::vector<std::string> SentenceIntoSeparateWords(std::string sentence);
};
#endif 