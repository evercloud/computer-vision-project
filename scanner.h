#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Six pipeline stages, one function each.

// Step 1: returns the four corners of the largest quadrilateral (the document).
// Returns empty vector if not found.
std::vector<cv::Point> detectDocumentContour(const cv::Mat& image,
                                              bool debug = false,
                                              const std::string& outputDir = ".");

// Step 2: sort the 4 corners into TL, TR, BR, BL order for getPerspectiveTransform.
// Returns empty vector on bad input.
std::vector<cv::Point2f> orderCorners(const std::vector<cv::Point>& corners);

// Step 3: compute the homography and warp the document to a flat rectangle.
cv::Mat warpDocument(const cv::Mat& image,
                     const std::vector<cv::Point2f>& orderedCorners);

// Step 4: convert the warped document to binary (black text on white background).
cv::Mat binarizeDocument(const cv::Mat& warpedImage);

// Step 5: count black pixels per row; result is a 1-D profile where
// peaks = text lines and valleys = inter-line gaps.
std::vector<int> computeHorizontalProjection(const cv::Mat& binaryImage);

// Step 6: detect gap/text transitions in the projection and crop each text line.
// Returns one sub-image per detected line.
std::vector<cv::Mat> segmentTextLines(const cv::Mat& binaryImage,
                                       const std::vector<int>& projection,
                                       int minLineHeight = 10,
                                       bool debug = false,
                                       const std::string& outputDir = ".");
