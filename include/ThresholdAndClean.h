/**
 * Xiaowei Zhang
 * 23SP
*/
#pragma once
#include <opencv2/core.hpp>

// bkg is black (0) and foreground is white (255)


template <typename T>
static void threshold(cv::Mat& src, cv::Mat& dst, const double t_up, const double t_low = 0) {
	for (auto i = 0; i < src.rows; ++i) {
		for (auto j = 0; j < src.cols; ++j) {
			if ((src.at<T>(i, j)[0] > t_up && src.at<T>(i, j)[1] > t_up && src.at<T>(i, j)[2] > t_up) ||
				(src.at<T>(i, j)[0] < t_low && src.at<T>(i, j)[1] < t_low && src.at<T>(i, j)[2] < t_low)) {
				dst.at<uchar>(i, j) = 0;
			}
			else { dst.at<uchar>(i, j) = 255; }
		}
	}
}


// Code for cleaning
 
static void helper(const cv::Mat& src, cv::Mat& dst, const int connectivity, const uchar c) {
	for (auto i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			bool doIt = false;
			if (src.at<uchar>(i, j) == c) { dst.at<uchar>(i, j) = src.at<uchar>(i, j); }
			// left side
			if (i > 0) {
				if (j > 0 && connectivity == 8 && src.at<uchar>(i - 1, j - 1) == c) { doIt = true; }
				if (src.at<uchar>(i - 1, j) == c) { doIt = true; }
				if (j < src.cols - 1 && connectivity == 8 && src.at<uchar>(i - 1, j + 1) == c) { doIt = true; }
			}
			// right side
			if (i < src.rows - 1) {
				if (j > 0 && connectivity == 8 && src.at<uchar>(i + 1, j - 1) == c) { doIt = true; }
				if (src.at<uchar>(i + 1, j) == c) { doIt = true; }
				if (j < src.cols - 1 && connectivity == 8 && src.at<uchar>(i + 1, j + 1) == c) { doIt = true; }
			}
			// top middle
			if (j > 0 && src.at<uchar>(i, j - 1) == c) { doIt = true; }
			// bottom middle
			if (j < src.cols - 1 && src.at<uchar>(i, j + 1) == c) { doIt = true; }
			// now erode/dilate if needed
			if (doIt) { dst.at<uchar>(i, j) = c; }
			else { dst.at<uchar>(i, j) = src.at<uchar>(i, j); }
		}
	}
}


static void erode(const cv::Mat& src, cv::Mat& dst, const int connectivity) { helper(src, dst, connectivity, 0); }


static void dilate(const cv::Mat& src, cv::Mat& dst, const int connectivity) { helper(src, dst, connectivity, 255); }


static void opening(cv::Mat& src, cv::Mat& dst, int erosionConn, int dilationConn) {
	cv::Mat temp(src.size(), CV_8UC1);
	erode(src, temp, erosionConn);
	dilate(temp, dst, dilationConn);
}

static void closing(cv::Mat& src, cv::Mat& dst, int dilationConn, int erosionConn) {
	cv::Mat temp(src.size(), CV_8UC1);
	dilate(src, temp, dilationConn);
	erode(temp, dst, erosionConn);
}


// Grassfire transform - extn

static void grassfireTransform(cv::Mat& src, cv::Mat& dst, int c, int connectivity) {
	// pass one
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (src.at<uchar>(i, j) == c) { dst.at<int>(i, j) = 0; }
			else {
				int m = INT32_MAX;
				if (i > 0) { m = std::min(m, dst.at<int>(i - 1, j) + 1); }
				else { m = std::min(m, 1); }
				if (j > 0) { m = std::min(m, dst.at<int>(i, j - 1) + 1); }
				else { m = std::min(m,  1); }

				if (connectivity == 8 && i > 0) {
					if (j > 0) { m = std::min(m, dst.at<int>(i - 1, j - 1) + 1); }
					if (j < src.cols - 1) { m = std::min(m, dst.at<int>(i - 1, j + 1) + 1); }
				}
				dst.at<int>(i, j) = m;
			}
		}
	}

	// pass 2
	for (int i = src.rows - 1; i > -1; --i) {
		for (int j = src.cols - 1; j > -1; --j) {
			if (src.at<uchar>(i, j) == c) { dst.at<int>(i, j) = 0; }
			else {
				int m = dst.at<int>(i, j);
				if (i < src.rows - 1) { m = std::min(m, dst.at<int>(i + 1, j) + 1); }
				else { m = std::min(m,  1); }
				if (j < src.cols - 1) { m = std::min(m, dst.at<int>(i, j + 1) + 1); }
				else { m = std::min(m,  1); }

				if (connectivity == 8 && i < src.rows - 1) {
					if (j < src.cols - 1) { m = std::min(m, dst.at<int>(i + 1, j + 1) + 1); }
					if (j > 0) { m = std::min(m, dst.at<int>(i + 1, j - 1) + 1); }
				}
				dst.at<int>(i, j) = m;
			}
		}
	}
}

static void grassfireOpen(cv::Mat& src, cv::Mat& dst, const int distToRemove, const int erosionConn = 4,
                          const int dilationConn = 8) {
	// erosion
	cv::Mat op(src.size(), CV_32SC1);
	cv::Mat temp(src.size(), CV_8UC1);
	grassfireTransform(src, op, 0, erosionConn);
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (op.at<int>(i, j) <= distToRemove) { temp.at<uchar>(i, j) = 0; }
			else { temp.at<uchar>(i, j) = src.at<uchar>(i, j); }
		}
	}

	// dilation
	cv::Mat cl(src.size(), CV_32SC1);
	grassfireTransform(temp, cl, 255, dilationConn);
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (cl.at<int>(i, j) <= distToRemove) { dst.at<uchar>(i, j) = 255; }
			else { dst.at<uchar>(i, j) = src.at<uchar>(i, j); }
		}
	}
}

static void grassfireClose(cv::Mat& src, cv::Mat& dst, const int distToRemove, const int dilationConn = 4,
                           const int erosionConn = 8) {
	// dilation
	cv::Mat cl(src.size(), CV_32SC1);
	cv::Mat temp(src.size(), CV_8UC1);
	grassfireTransform(src, cl, 255, dilationConn);
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (cl.at<int>(i, j) <= distToRemove) { temp.at<uchar>(i, j) = 255; }
			else { temp.at<uchar>(i, j) = src.at<uchar>(i, j); }
		}
	}

	// erosion
	cv::Mat op(src.size(), CV_32SC1);
	grassfireTransform(src, op, 0, erosionConn);
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (op.at<int>(i, j) <= distToRemove) { dst.at<uchar>(i, j) = 0; }
			else { dst.at<uchar>(i, j) = src.at<uchar>(i, j); }
		}
	}
}

static void grassfireClean(cv::Mat& src, cv::Mat& dst, const int distToRemove) {
	cv::Mat temp(src.size(), CV_8UC1);
	grassfireOpen(src, temp, distToRemove);
	grassfireClose(temp, dst, distToRemove);
}
