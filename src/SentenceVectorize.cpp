#include "SentenceVectorize.h"


std::vector<std::string> WordsVectorize::SentenceIntoSeparateWords(const std::vector<std::string>& SQL) {
	std::vector<std::string> res;
	for (auto sentence : SQL) {
		std::istringstream s(sentence);
		std::string word;
		while (s >> word) {
			res.push_back(word);
		}
	}

	return res;
}

std::vector<std::string> WordsVectorize::SentenceIntoSeparateWords(std::string sentence,int &totalWords) {
	std::vector<std::string> res;
		std::istringstream s(sentence);
		std::string word;
		while (s >> word) {
			res.push_back(word);
			totalWords++;
		}

	return res;
}

std::vector<std::string> WordsVectorize::SentenceIntoSeparateWords(std::string sentence) {
	std::vector<std::string> res;
	std::istringstream s(sentence);
	std::string word;
	while (s >> word) {
		res.push_back(word);
	}

	return res;
}


void WordsVectorize::MapIndexInit(std::map<std::string, int> numOfWordsInSentenceInDataset,std::vector<std::string> vec) {
	for (auto n : vec) {
		this->numOfWordsInSentenceInDataset[n] = 0;
	}
}

void WordsVectorize::MapIndexInit(std::map<std::string, double> numOfWordsInSentenceInDataset, std::vector<std::string> vec) {
	for (auto n : vec) {
		this->numOfWordsInSentenceInDataset[n] = 0.0;
	}
}

void WordsVectorize::FindsNumOfWordsInDatasetSentences(const std::vector<std::string>& methods, const std::vector<std::string>& pages,
	const std::vector<std::string>& protocol,
	const std::vector<std::string>& agent,
	const std::vector<std::string>& SQL, const std::vector<std::string>& XSS, const std::vector<std::string>& Win,
	std::vector<std::string> dataset) {
	MapIndexInit(numOfWordsInSentenceInDataset,methods);
	MapIndexInit(numOfWordsInSentenceInDataset,pages);
	MapIndexInit(numOfWordsInSentenceInDataset,protocol);
	MapIndexInit(numOfWordsInSentenceInDataset,agent);
	MapIndexInit(numOfWordsInSentenceInDataset,SQL);
	MapIndexInit(numOfWordsInSentenceInDataset,XSS);
	MapIndexInit(numOfWordsInSentenceInDataset,Win);

	for (auto sentence: dataset) {
		std::istringstream s(sentence);
		std::string word;
		while (s >> word) {
			numOfWordsInSentenceInDataset[word]++;
		}
	}
}

void WordsVectorize::calculIDF() {
	int i = 0;
	for (auto words: numOfWordsInSentenceInDataset) {
		idf[i].word = words.first;
		idf[i].IDF = log10(datasetSize / words.second);
		i++;
	}
}

void WordsVectorize::SaveIDFToFile() {
	std::ofstream fout;
	fout.open(path, std::ios::binary);
	for (int i = 0; i < numOfWordsInSentenceInDataset.size(); i++) {
		fout.write((char*)&idf[i], sizeof(WordsIDF));
	}
}

void WordsVectorize::ReadIDFFromFile() {
	WordsIDF* wordIdf;
	wordIdf = new WordsIDF[uniqueWordsAmount];
	std::ifstream fin;
	fin.open(path, std::ios::binary);
		for (int i = 0; i < numOfWordsInSentenceInDataset.size(); i++) {
			fin.read((char*)&wordIdf[i], sizeof(WordsIDF));
		}
		IDF.clear();
		for (int i = 0; i < uniqueWordsAmount;i ++) {
			IDF[wordIdf[i].word] = wordIdf[i].IDF;
		}
}


std::map<std::string, double> WordsVectorize::TF(std::string sentence) {
	int totalWords;
	std::vector<std::string> words = SentenceIntoSeparateWords(sentence,totalWords);
	std::map<std::string, double> res;
	MapIndexInit(res, words);
	for (auto &n : words) {
		res[n]++;
	}

	for (auto &n : res) {
		n.second = n.second / totalWords;
	}
	return res;
}

std::vector<double> WordsVectorize::TFxIDF(std::string sentence,const unsigned int& inputNeuronsAmount) {

	std::vector<double> res(inputNeuronsAmount, 0.0);
	std::map<std::string, double> sentenceTF = TF(sentence);

	for (auto n : sentenceTF) {
		res.push_back(n.second);
	}
	int i = 0;
	for (auto n : IDF) {
		res[i] = res[i] * n.second;
		i++;
	}
	return res;
}
