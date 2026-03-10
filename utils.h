#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Small utilities shared across the pipeline.

// Save an image to disk with a simple console confirmation.
void saveImage(const std::string& path, const cv::Mat& image);

// Draw a filled polygon outline on top of a copy of 'image' and return it.
// Used to visualise the detected document quadrilateral during debugging.
cv::Mat drawContourDebug(const cv::Mat& image, const std::vector<cv::Point>& contour);
