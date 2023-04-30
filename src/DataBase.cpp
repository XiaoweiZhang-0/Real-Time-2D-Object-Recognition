/**
 * Xiaowei Zhang
 * 23SP
*/

#include "DataBase.hpp"

DataBase :: DataBase()
{
    
}

DataBase :: ~DataBase(){
    std::cout << "File closed!" << std::endl;
    if(dbfile.is_open())
        dbfile.close();
}

int DataBase::storeFeatureVectorInDB(std::vector<double> &features, std::string &label)
{
    if(!dbfile.is_open())
        std::cout << "DB file not open!" << std::endl;
    else
    {
        dbfile << label;
        for (auto f : features) {
            dbfile << " " << f;
            std::cout<<"F:" << f; 
        }
        dbfile << std::endl;
        std::cout<<"F:" << std::endl;
    }
    return 0;
}

int DataBase::getMatchFromDB(std::vector<double> &features, std::string &label)
{
    dbfile.clear();
    dbfile.seekg(0, std::ios::beg);
    std::vector<double> tempFeatures;
    double currentDistance = std::numeric_limits<double>::infinity();
    double newDistance;
    double f1;
    std::string tempLabel;
    
    std::string line;
    while (std::getline(dbfile, line))
    {
        tempFeatures.clear();
        std::stringstream lineStream(line);
        lineStream >> tempLabel;
        while(lineStream >> f1)
        {
            tempFeatures.push_back(f1);
        }
        newDistance = getEuclideanDistance(features, tempFeatures);
        //std::cout << "Distance:" << newDistance <<std::endl;
        if(currentDistance > newDistance)
        {
            label.assign(tempLabel);
            //std::cout<<"Current Label:" << label << std::endl;
            currentDistance = newDistance;
        }
    }
    
    return 0;
}

double DataBase::getEuclideanDistance(std::vector<double> feature1, std::vector<double> feature2)
{
    std::vector<double> feature1new;
    std::vector<double> feature2new;
    for (int i = 0; i < feature1.size(); i++)
    {
        feature1[i] = (feature1[i] - mean[i])/stdDev[i];
        feature2[i] = (feature2[i] - mean[i])/stdDev[i];
    }
    return cv::norm(feature1,feature2);
}

int DataBase::setMeanAndStdDevForEachFeature()
{
    dbfile.clear();
    dbfile.seekg(0, std::ios::beg);
    std::vector<std::vector<double>> features;
    
    double f1;
    double sum;
    double sq_sum;
    double stddev;
    std::string tempLabel;
    std::string line;
    bool firstLine = true;
    while (std::getline(dbfile, line))
    {
        std::stringstream lineStream(line);
        lineStream >> tempLabel;
        int i=0;
        
        while(lineStream >> f1)
        {
            if(firstLine)
            {
                std::vector<double> temp;
                temp.push_back(f1);
                features.push_back(temp);
                temp.clear();
            }
            else
                features[i++].push_back(f1);
        }
        firstLine = false;
    }

    //Calculate mean
    for (int i = 0; i<features.size() ; i++)
    {
        sum = std::accumulate(features[i].begin(), features[i].end(), 0.0);
        double tempMean = sum/features[i].size();
        mean.push_back(tempMean);
    }
    
    std::cout << std::endl;
    std::vector<double> diff1;
    for (int i = 0; i<features.size() ; i++)
    {
        std::vector<double> diff1(features[i].size());
        std::transform(features[i].begin(), features[i].end(), diff1.begin(),
                       std::bind(std::minus<double>(),std::placeholders::_1, mean[i]));
        sq_sum = std::inner_product(diff1.begin(), diff1.end(), diff1.begin(), 0.0);
        stddev = std::sqrt(sq_sum / features[i].size());
        stdDev.push_back(stddev);
        diff1.clear();
    }
    std::cout << std::endl;
    std::cout << "stdDev:" ;
    for (int i = 0; i<stdDev.size() ; i++)
    {
        std::cout << stdDev[i] << " " ;
    }
    return 0;
}

int DataBase::fileDB()
{
    if(dbfile.is_open())
    {
        dbfile.close();
        return 1;
    }
    return 0;
}

int DataBase::openDB(char mode)
{
    if(mode == 'w')
        dbfile.open("FeatureVector_DB.txt", std::ios::out);
    else
        dbfile.open("FeatureVector_DB.txt", std::ios::in);
    if(!dbfile.is_open())
        std::cout << "DB file not open!" << std::endl;
    
    return 0;
}

int DataBase::getKNNMatchFromDB(std::vector<double> &features, std::priority_queue <std::pair<int,std::string>> &kmatches, int K)
{
    dbfile.clear();
    dbfile.seekg(0, std::ios::beg);
    std::vector<double> tempFeatures;
    double currentDistance = std::numeric_limits<double>::infinity();
    double newDistance;
    double f1;
    std::string tempLabel;
    
    std::string line;
    while (std::getline(dbfile, line))
    {
        tempFeatures.clear();
        std::stringstream lineStream(line);
        lineStream >> tempLabel;
        while(lineStream >> f1)
        {
            tempFeatures.push_back(f1);
        }
        newDistance = getEuclideanDistance(features, tempFeatures);
        //std::cout << "Distance:" << newDistance <<std::endl;
        
        if(kmatches.size() < K)
        {
            kmatches.push(make_pair(newDistance,tempLabel));
        }
        else if(currentDistance > newDistance)
        {
            currentDistance = newDistance;
            kmatches.pop();
            kmatches.push(make_pair(newDistance,tempLabel));
        }
    }
    
    return 0;
}
