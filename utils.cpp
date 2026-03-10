#include "utils.h"
#include <iostream>


void saveImage(const std::string& path, const cv::Mat& image)
{
    bool ok = cv::imwrite(path, image);
    if (ok)
        std::cout << "[saved] " << path << "\n";
    else
        std::cerr << "[error] Could not save: " << path << "\n";
}

cv::Mat drawContourDebug(const cv::Mat& image, const std::vector<cv::Point>& contour)
{
    cv::Mat display = image.clone();

    // polylines expects a vector-of-vectors
    std::vector<std::vector<cv::Point>> contours = { contour };
    cv::polylines(display, contours, /*isClosed=*/true, cv::Scalar(0, 255, 0), 3);

    // Also mark each corner with a small circle so we can verify ordering
    for (const auto& pt : contour)
        cv::circle(display, pt, 8, cv::Scalar(0, 0, 255), -1);

    return display;
}
