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


void WordsVectorize::MapIndexInit(std::map<std::string, int> &tempMap,std::vector<std::string> vec) {
	for (const auto &n : vec) {
		tempMap[n] = 0;
	}
}

void WordsVectorize::MapIndexInit(std::map<std::string, double> &tempMap, std::vector<std::string> vec) {
	for (const auto &n : vec) {
		tempMap[n] = 0.0;
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

	for (const auto &sentence: dataset) {
		std::istringstream s(sentence);
		std::string word;
		std::set<std::string> unique;
		while (s >> word) {
			unique.insert(word);
		}
		for(auto &w:unique){
			numOfWordsInSentenceInDataset[w]++;
		}
	}
}

void WordsVectorize::calculIDF() {
	int i = 0;
	idf = new WordsIDF[numOfWordsInSentenceInDataset.size()];
	for (const auto &words: numOfWordsInSentenceInDataset) {
		idf[i].word = new char[words.first.size() + 1];
		FromStringToChar(words.first, idf[i].word, words.first.size());
		idf[i].word[words.first.size()] = '\0';
		if (words.second != 0) {
			idf[i].IDF = log10((double)datasetSize / words.second);
		}
		else {
			idf[i].IDF = log10((double)datasetSize / 1);
		}
		idf[i].size = words.first.size();
		IDF[words.first] = idf[i].IDF;
		i++;
	}
}

void WordsVectorize::SaveIDFToFile() {
	std::ofstream fout;
	fout.open(WordsVectorize::path, std::ios::binary);
	for (int i = 0; i < numOfWordsInSentenceInDataset.size(); i++) {
		fout.write((char*)&idf[i].size, sizeof(int));
		fout.write(idf[i].word, idf[i].size);
		fout.write((char*)&idf[i].IDF, sizeof(double));
	}
}

void WordsVectorize::ReadIDFFromFile() {
	std::ifstream fin;
	IDF.clear();
	fin.open(WordsVectorize::path, std::ios::binary);
		for (int i = 0; i < numOfWordsInSentenceInDataset.size(); i++) {
			int size = 0;
			fin.read((char*)&size, sizeof(int));
			std::vector<char> wordBuffer(size + 1, '\0');
			fin.read(wordBuffer.data(), size);
			std::string word(wordBuffer.data());
			double wordIDF;
			fin.read((char*)&wordIDF, sizeof(double));
			IDF[word] = wordIDF;

		}


}


std::map<std::string, double> WordsVectorize::TF(std::string sentence) {
	int totalWords = 0;
	std::vector<std::string> words = SentenceIntoSeparateWords(sentence,totalWords);
	std::map<std::string, double> res;
	MapIndexInit(res, words);
	for (const auto &n : words) {
		res[n]++;
	}

	for (auto &n : res) {
		n.second = n.second / totalWords;
	}
	return res;
}

std::vector<double> WordsVectorize::TFxIDF(std::string sentence,const unsigned int& inputNeuronsAmount) {

	std::vector<double> res(inputNeuronsAmount,0);
	std::map<std::string, double> sentenceTF = TF(sentence);
	int i = 0;
	for (const auto &n : IDF) {
		if (i >= inputNeuronsAmount) break;
		std::string currrentWord = n.first;
		double currentIdf = n.second;
		auto it = sentenceTF.find(currrentWord);
		if (it != sentenceTF.end()) {
			res[i] = it->second * currentIdf;
			//std::cout << "TF-IDF: " << res[i] << std::endl;
		}
		else {
			res[i] = 0.0;
		}

		i++;
	}
	
	return res;
}

void WordsVectorize::FromStringToChar(std::string sentence,char* arr,int size) {
	if (size != sentence.size()) return;
	else {
		for (int i = 0; i < size; i++) {
			arr[i] = sentence[i];
		}
	}
}
void WordsVectorize::FromCharToString(char* arr, std::string sentence,int size) {
	if (size != sentence.size()) return;
	else {
		for (int i = 0; i < size; i++) {
			sentence[i] = arr[i];
		}
	}
}

void WordsVectorize::ShowIDF() {
	std::cout << "IDF:" << std::endl;
	for (auto n : IDF) {
		std::cout <<"Word" << n.first << "\tIDF" << n.second << std::endl;
	}
}

