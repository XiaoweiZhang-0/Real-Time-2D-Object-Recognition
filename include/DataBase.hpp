/**
 * Xiaowei Zhang
 * 23SP
*/

#ifndef DataBase_hpp
#define DataBase_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <numeric>
#include <opencv2/core.hpp>
#include <queue>

class DataBase
{
private:
    std::fstream dbfile;
    std::vector<double> mean;
    std::vector<double> stdDev;
    
    double getEuclideanDistance(std::vector<double> feature1, std::vector<double> feature2);
public:
    DataBase();
    ~DataBase();
    int setMeanAndStdDevForEachFeature();
    int storeFeatureVectorInDB(std::vector<double> &features, std::string &label);
    int getMatchFromDB(std::vector<double> &features, std::string &label);
    int fileDB();
    int openDB(char mode);
    
    int getKNNMatchFromDB(std::vector<double> &features, std::priority_queue <std::pair<int,std::string>> &kmatches, int K);
};

#endif /* DataBase_hpp */


