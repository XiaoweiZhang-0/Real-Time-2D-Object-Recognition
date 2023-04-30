/**
 * Xiaowei Zhang
 * 23SP
*/
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// ref: https://docs.opencv.org/3.4/de/d01/samples_2cpp_2connected_components_8cpp-example.html
// task 2 - only selects and colors top N components
static void segmentAndColorRegions(cv::Mat& src, cv::Mat& dst, int connectivity = 4, int N = 1) {
	srand(1);
	cv::Mat labels;
	cv::Mat stats;
	cv::Mat centroids;
	int numLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids);
	std::vector<cv::Vec3b> colors(numLabels);
	colors[0] = cv::Vec3b(0, 0, 0); // black for the bkg
	std::vector<int> sortedArea;
	for (int i = 0; i < numLabels; ++i) { sortedArea.push_back(stats.at<int>(i, cv::CC_STAT_AREA)); }
	std::sort(sortedArea.rbegin(), sortedArea.rend());
	for (int i = 1; i < numLabels; ++i) {
		if (N >= numLabels || stats.at<int>(i, cv::CC_STAT_AREA) >= sortedArea.at(N)) {
			colors[i] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
		}
		else { colors[i] = cv::Vec3b(0, 0, 0); }
	}

	for (auto i = 0; i < src.rows; ++i) {
		for (auto j = 0; j < src.cols; ++j) { dst.at<cv::Vec3b>(i, j) = colors[labels.at<int>(i, j)]; }
	}
}


static std::vector<std::vector<std::pair<int, int>>>* getListOfCoordsForEachRegion(cv::Mat& labels) {
	auto* listOfCoords = new std::vector<std::vector<std::pair<int, int>>>;
	for (int i = 0; i < labels.rows; ++i) {
		for (int j = 0; j < labels.cols; ++j) {
			if (labels.at<int>(i, j) >= listOfCoords->size()) {
				std::vector<std::pair<int, int>> vec;
				// flip i and j because y corresponds to the row and x corresponds to the col
				vec.emplace_back(j, i);
				listOfCoords->push_back(vec);
			}
			else {
				// used .emplace_back instead of .push_back(std::pair<int, int>(i, j))
				listOfCoords->at(labels.at<int>(i, j)).emplace_back(j, i);
			}
		}
	}
	return listOfCoords;
}


// Extn - Self implemented connected components code using union find
/**
 * Amortized linear time Union Find data structure
 */
class UnionFind {
private:
	int arr[1000];
	int size = 1000;

public:

	UnionFind() { for (int i = 0; i < size; ++i) { arr[i] = i; } }

	void Union(int x, int y) {
		std::vector<int> store;
		while (x != arr[x]) {
			store.push_back(x);
			x = arr[x];
		}
		while (y != arr[y]) {
			store.push_back(y);
			y = arr[y];
		}
		arr[x] = y;
		for (auto a : store) { arr[a] = y; }
	}

	int Find(int x) {
		std::vector<int> store;
		while (x != arr[x]) {
			store.push_back(x);
			x = arr[x];
		}
		for (auto a : store) { arr[a] = x; }
		return x;
	}
};

class Segmentation {
private:
	cv::Mat* labels;
	int N;
	int ignoreArea;
	int numLabels;

public:
	Segmentation(cv::Mat* img, int ignoreArea = 1000, int N = 1) : ignoreArea(ignoreArea), N(N) {
		labels = new cv::Mat(img->size(), CV_32SC1, cv::Scalar(0));
		findConnectedComponents(img);
	}

	void getListOfCoordsForEachRegion(std::vector<std::vector<std::pair<int, int>>>* listOfCoords) {
		for (int i = 0; i < labels->rows; ++i) {
			for (int j = 0; j < labels->cols; ++j) {
				if (labels->at<int>(i, j) == 0) { continue; }
				if (labels->at<int>(i, j) > listOfCoords->size()) {
					std::vector<std::pair<int, int>> vec;
					vec.emplace_back(j, i);
					listOfCoords->push_back(vec);
				}
				else {
					listOfCoords->at(labels->at<int>(i, j) - 1).emplace_back(j, i);
				}
			}
		}
	}


	void colorRegions(cv::Mat& dst) {
		srand(1);

		std::vector<cv::Vec3b> colors(numLabels + 1);
		colors[0] = cv::Vec3b(0, 0, 0); // black for the bkg
		for (int i = 1; i < numLabels + 1; ++i) {
			colors[i] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
		}

		for (auto i = 0; i < dst.rows; ++i) {
			for (auto j = 0; j < dst.cols; ++j) { dst.at<cv::Vec3b>(i, j) = colors[labels->at<int>(i, j)]; }
		}
	}

private:
	// with a connectivity of 4
	void findConnectedComponents(cv::Mat* img) {
		UnionFind uf;
		int regions = 0;
		int ignoreBorder = 1;
		for (int i = ignoreBorder; i < img->rows - ignoreBorder; ++i) {
			for (int j = ignoreBorder; j < img->cols - ignoreBorder; ++j) {
				if (img->at<uchar>(i, j) == 0) { continue; }
				if (labels->at<int>(i - 1, j) == 0 && labels->at<int>(i, j - 1) == 0) {
					labels->at<int>(i, j) = ++regions;
				}
				else if (labels->at<int>(i - 1, j) == 0) { labels->at<int>(i, j) = labels->at<int>(i, j - 1); }
				else if (labels->at<int>(i, j - 1) == 0) { labels->at<int>(i, j) = labels->at<int>(i - 1, j); }
				else {
					uf.Union(labels->at<int>(i - 1, j), labels->at<int>(i, j - 1));
					labels->at<int>(i, j) = labels->at<int>(i - 1, j);
				}
			}
		}
		auto* areas = new std::vector<int>(regions + 1, 0);

		for (int i = 1; i < img->rows - 1; ++i) {
			for (int j = 1; j < img->cols - 1; ++j) {
				if (img->at<uchar>(i, j) == 0) { continue; }
				labels->at<int>(i, j) = uf.Find(labels->at<int>(i, j));
				areas->at(labels->at<int>(i, j)) += 1;
			}
		}

		// removing the not needed components		
		std::vector<int> sortedAreas(*areas);
		std::sort(sortedAreas.rbegin(), sortedAreas.rend());

		std::vector<int> labelMap(1000, 0);
		regions = 0;

		for (auto i = 0; i < labels->rows; ++i) {
			for (auto j = 0; j < labels->cols; ++j) {
				auto l = labels->at<int>(i, j);
				if (labelMap[l] == 0 && regions < N && areas->at(l) > ignoreArea && areas->at(l) > sortedAreas.at(N)) {
					labelMap[l] = ++regions;
				}
				labels->at<int>(i, j) = labelMap[l];
			}
		}
		numLabels = regions;
		delete areas;
	}
};
