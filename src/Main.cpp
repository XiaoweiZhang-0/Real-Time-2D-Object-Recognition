/**
 * Xiaowei Zhang
 * 23SP
*/
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include "Segmentation.h"
#include "ThresholdAndClean.h"
#include "Features.h"
#include "DataBase.hpp"

#define CVUI_IMPLEMENTATION
#include "cvui.h"

using namespace std;
using namespace cv;
namespace fs = std::__fs::filesystem;


#define WINDOW_NAME "Project3"
vector<Mat> readImagesFromFolder(const string& folderPath) {
    vector<Mat> images;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        string imagePath = entry.path().string();
        if(imagePath.substr(imagePath.length()-3, 3) != "jpg"){
            continue;
        }
        Mat img = imread(imagePath, IMREAD_COLOR);
        if (!img.empty()) {
            images.push_back(img);
        } else {
            cout << "Failed to load image: " << imagePath << endl;
        }
    }
    return images;
}

int main(int argc, char *argv[]) {
    cv::VideoCapture* capDev;
    capDev = new cv::VideoCapture(0);
    if (!capDev->isOpened()) {
        std::cout << "Can't open Video device" << std::endl;
        return -1;
    }

    cv::Size refS((int)capDev->get(cv::CAP_PROP_FRAME_WIDTH), (int)capDev->get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "Expected size: " << refS.width << " " << refS.height << std::endl;
    char mode = 'r';  //Default is test mode
    cv::Mat frame;
    cv::Mat segm;
    int scNum = 0;
    // std::string tr_label;
    bool detectObject = false;
    bool trainingMode = false;
    DataBase db;
    int noOfObjectsToDetect = 1;
    std::vector<double> features;
    std::string tr_label;
    cv::Mat mainWindow = cv::Mat(700, 1000, CV_8UC3);
    mainWindow = cv::Scalar(49, 52, 49);
    // Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
    cvui::init(WINDOW_NAME);
    
    // while (true) {
    //create a folder called thresholdedImages
    std::__fs::filesystem::create_directory("ReportFiles/thresholdedImages");
    std::__fs::filesystem::create_directory("ReportFiles/cleanedImages");
    std::__fs::filesystem::create_directory("ReportFiles/segmentedImages");
    std::__fs::filesystem::create_directory("ReportFiles/coloredImages");

    while(true){
    // vector<Mat> images = readImagesFromFolder("trainingSet");
    // for(auto i :images){

        //Process frame
        features.clear();
        
        *capDev >> frame;
        // frame = i;
        cv::resize(frame, frame, cv::Size(), 0.20, 0.20);
        // threshold the video
        cv::Mat thresh(frame.size(), CV_8UC1);
        threshold<cv::Vec3b>(frame, thresh, 0.5*255);

        // cleanup the video
        // cv::Mat temp(frame.size(), CV_8UC1);
        cv::Mat cleaned(frame.size(), CV_8UC1);
        // opening(thresh, temp, 8, 4);
        // closing(temp, cleaned, 8, 4);
        grassfireClean(thresh, cleaned, 3);

        Segmentation seg(&cleaned, 100, noOfObjectsToDetect);
        std::vector<std::vector<std::pair<int, int>>>*  listOfCoords = new std::vector<std::vector<std::pair<int, int>>>;
        seg.getListOfCoordsForEachRegion(listOfCoords);

        cv::Mat colored(frame.size(), CV_8UC3);
        seg.colorRegions(colored);
        // cv::imwrite("ReportFiles/segmentedImages/segmented_" + std::to_string(scNum) + ".jpg", colored);
        // segm = colored;

        int count = 0;
        for (auto regionCoords : *listOfCoords) {
            
            //getFeatures(regionCoords, features);
            auto* bb = getFeatures(regionCoords, features);

            std::vector<cv::Point> points;
            for (auto [x,y] : *bb) {
                cv::Point p(x, y);
                points.push_back(p);
            }
            
            cv::line(colored, points.at(0), points.at(1), cv::Scalar(0, 255, 0), 2);
            cv::line(colored, points.at(1), points.at(2), cv::Scalar(0, 255, 0), 2);
            cv::line(colored, points.at(2), points.at(3), cv::Scalar(0, 255, 0), 2);
            cv::line(colored, points.at(3), points.at(0), cv::Scalar(0, 255, 0), 2);

            cv::line(colored, points.at(4), points.at(5), cv::Scalar(0, 255, 0), 2);
            
            cv::circle(colored, points.at(6), 2, cv::Scalar(0, 255, 255), 2);
            
            if( trainingMode == false && detectObject == true )
            {
                
                db.getMatchFromDB(features,tr_label);
                std::cout << "Detected Object" << count++ << ":" << tr_label << std::endl;
                cvui::printf(colored, points.at(0).x,points.at(0).y, 0.4, 0xff0000, "%s", tr_label.c_str());
                features.clear();
            }
        }

        cvui::printf(mainWindow, 10, 220, 0.4, 0xff0000, "No of objects to detect:");
        cvui::trackbar(mainWindow, 10, 230, 200, &noOfObjectsToDetect, (int)1, (int)3);
        
        if (cvui::button(mainWindow, 10, 10,100,30, "Train Mode"))
        {
            // The button was clicked, current task set to this.
            noOfObjectsToDetect = 1;
            trainingMode = true;
            std::cout << "Entered Training Mode. New database file will be created!" << std::endl;
            db.openDB('w');
        }
        if (cvui::button(mainWindow, 10, 40,100,30, "Save Label"))
        {
            // The button was clicked, current task set to this.
            if(trainingMode)
            {
                std::cout << "Enter label Name:" << std::endl;
                std::cin >> tr_label;
                db.storeFeatureVectorInDB(features,tr_label);
            }
            else
                std::cout << "Start training mode first!" << std::endl;
        }
        if (cvui::button(mainWindow, 10, 70,100,30, "Test Mode"))
        {
            // The button was clicked, current task set to this.
            if(db.fileDB())
                std::cout << "Training mode finished!" << std::endl;
            db.openDB('r');
            std::cout << "In test Mode! Dataset ready!" << std::endl;
            db.setMeanAndStdDevForEachFeature();
            noOfObjectsToDetect = 2;
            trainingMode = false;
        }
        if (cvui::button(mainWindow, 10, 100,100,30, "Detect"))
        {
            // The button was clicked, current task set to this.
            detectObject = true;
        }
        if (cvui::button(mainWindow, 10, 130,100,30, "Stop Detection!"))
        {
            // The button was clicked, current task set to this.
            detectObject = false;
        }
        delete listOfCoords;
        
        cvui::printf(mainWindow, 210, 30, 0.5, 0xff0000, "Original Video");
        cvui::image(mainWindow, 210, 50, frame);
        cvui::printf(mainWindow, 610, 30, 0.5, 0xff0000, "Thresholded Video");
        cv::cvtColor(thresh, thresh, cv::COLOR_GRAY2RGB);
        cvui::image(mainWindow, 610, 50, thresh);
        cvui::printf(mainWindow, 210, 330, 0.5, 0xff0000, "Cleaned Video");
        cv::cvtColor(cleaned, cleaned, cv::COLOR_GRAY2RGB);
        cvui::image(mainWindow, 210, 350, cleaned);
        cvui::printf(mainWindow, 610, 330, 0.5, 0xff0000, "Colored Video");
        cvui::image(mainWindow, 610, 350, colored);
        cvui::update();
        
        cvui::imshow(WINDOW_NAME, mainWindow );
        // Check if ESC key was pressed
        if (cv::waitKey(20) == 27) {
            break;
        }
        
    }

    delete capDev;
    return 0;
}

