/**
 * Xiaowei Zhang
 * 23SP
*/
#pragma once

#include <vector>
#define M_PI 3.14159265358979323846

template <typename T>
static T min(T a, T b) { return a > b ? b : a; }

template <typename T>
static T max(T a, T b) { return a < b ? b : a; }

// define a generic moment function
static double moment(std::vector<std::pair<int, int>>& vector, int p, int q) {
	auto m = 0.0;
	for (auto coord : vector) {
		auto x = coord.first;
		auto y = coord.second;
		m += pow(x, p) * pow(y, q);
	}
	return m;
}

// centroid function
static std::pair<double, double> findCentroid(std::vector<std::pair<int, int>>& vector, double m00 = -1) {
	if (m00 == -1) { m00 = moment(vector, 0, 0); }
	return std::pair<double, double>(moment(vector, 1, 0) / m00, moment(vector, 0, 1) / m00);
}

// a general central moment
static double centralMoment(std::vector<std::pair<int, int>>& vector, int p, int q, double m00 = -1) {
	auto m = 0.0;
	auto c = findCentroid(vector);
	auto c_x = c.first;
	auto c_y = c.second;
	for (auto coord : vector) {
		auto x = coord.first;
		auto y = coord.second;
		m += pow(x - c_x, p) * pow(y - c_y, q);
	}
	if (m00 == -1) { m00 = moment(vector, 0, 0); }
	return m / m00;
}

// orientation of the central axis
static double findAlpha(std::vector<std::pair<int, int>>& vector) {
	return 0.5 * atan2(2 * centralMoment(vector, 1, 1),
	                   centralMoment(vector, 2, 0) - centralMoment(vector, 0, 2));
}

// Feature 1: second moment about the central axis
static double mu22Alpha(std::vector<std::pair<int, int>>& vector, double alpha, double m00 = -1) {
	auto m = 0.0;
	auto c = findCentroid(vector);
	auto c_x = c.first;
	auto c_y = c.second;
	auto beta = alpha + M_PI / 2;
	for (const auto [x, y] : vector) { m += pow((x - c_x) * sin(beta) + (y - c_y) * cos(beta), 2); }
	if (m00 == -1) { m00 = moment(vector, 0, 0); }
	return m / m00;
}

// project points to the new co-ordinate system with the centroid as the origin
template <typename T>
static void projectPoints(std::vector<std::pair<T, T>>& points, std::vector<std::pair<double, double>>& projected,
                          double alpha, std::pair<T, T> centroid) {
	auto c_x = centroid.first;
	auto c_y = centroid.second;
	for (const auto [x, y] : points) {
		projected.push_back({
			cos(alpha) * (1.0 * x - c_x) + sin(alpha) * (1.0 * y - c_y),
			-sin(alpha) * (1.0 * x - c_x) + cos(alpha) * (1.0 * y - c_y)
		});
	}
}

// get the dims of bounding box [min_x, miny_y, max_x, max_y]
template <typename T>
static void boundingBoxDims(std::vector<std::pair<T, T>>& points, T* bb) {
	for (const auto [x, y] : points) {
		bb[0] = min<T>(x, bb[0]);
		bb[1] = min<T>(y, bb[1]);
		bb[2] = max<T>(x, bb[2]);
		bb[3] = max<T>(y, bb[3]);
	}
}

// function which creates projects back the bounding box corners and the end points of a line segment for the central axis.
static void getOutputPoints(double boundingBoxDims[4], std::vector<std::pair<int, int>>* boundingBoxCorners,
                              double alpha,
                              std::pair<int, int> centroid) {
	std::vector<std::pair<double, double>> points;
	points.push_back({boundingBoxDims[2], boundingBoxDims[3]});
	points.push_back({boundingBoxDims[2], boundingBoxDims[1]});
	points.push_back({boundingBoxDims[0], boundingBoxDims[1]});
	points.push_back({boundingBoxDims[0], boundingBoxDims[3]});
	points.push_back({ boundingBoxDims[0] - 5, 0 });
	points.push_back({ boundingBoxDims[2] + 5, 0 });
	std::vector<std::pair<double, double>> projectedBack;
	auto c_x = centroid.first;
	auto c_y = centroid.second;
	projectPoints<double>(points, projectedBack, -alpha, {0,0});
	for (auto [x, y] : projectedBack) { boundingBoxCorners->push_back({x + c_x, y + c_y}); }
}

// creates a normalized bucketed histogram of the number of pixels when projected to the X and Y axes
template <typename T>
static void normalizedHistogramOfXAndY(std::vector<std::pair<T, T>> points, std::vector<double>& histogram,
                                       double boundingBoxDims[4],int numBuckets) {
	for (auto [x, y] : points) {
		auto a = static_cast<int>((1.0 * numBuckets) * ((1.0 * x - boundingBoxDims[0]) / (boundingBoxDims[2] -
			boundingBoxDims[0])));
		histogram[min<double>(numBuckets - 1,
		                      numBuckets * (1.0 * x - boundingBoxDims[0]) / (boundingBoxDims[2] - boundingBoxDims[0]
		                      ))] += 1;
		histogram[min<double>(numBuckets - 1,
		                      numBuckets * (1.0 * y - boundingBoxDims[1]) / (boundingBoxDims[3] - boundingBoxDims[1]
		                      )) + numBuckets] += 1;
	}
	// normalize
	auto sumX = 0.0;
	auto sumY = 0.0;
	for (int i = 0; i < numBuckets; ++i) {
		sumX += histogram[i];
		sumY += histogram[numBuckets + i];
	}
	for (int i = 0; i < numBuckets; ++i) {
		histogram[i] /= sumX;
		histogram[numBuckets + i] /= sumY;
	}
}


// then the function which combines it all together and gets the feature
// also returns the corner points of the bounding box, end points of a line segment for the central axis and the centroid (in that order).
static std::vector<std::pair<int, int>>* getFeatures(std::vector<std::pair<int, int>>& points,
                                                     std::vector<double>& features) {
	auto alpha = findAlpha(points);
	auto m00 = moment(points, 0, 0);
	auto centroid = findCentroid(points, m00);
	// mu 2 2 alpha
	features.push_back(mu22Alpha(points, alpha, m00));

	std::vector<std::pair<double, double>> projected;
	projectPoints<int>(points, projected, alpha, centroid);

	// ht/wd of the bounding box
	double bb[4] = { 0,0,0,0 };
	boundingBoxDims<double>(projected, bb);
	auto bb_h = bb[2] - bb[0];
	auto bb_w = bb[3] - bb[1];
	features.push_back((bb_h) / (bb_w));

	// % of the box filled
	features.push_back(m00 / (bb_h * bb_w));

	auto* outputPoints = new std::vector<std::pair<int, int>>;
	getOutputPoints(bb, outputPoints, alpha, centroid);
	outputPoints->push_back({ centroid.first, centroid.second });
	return outputPoints;
}
