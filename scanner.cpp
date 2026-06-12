#include "scanner.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>

// Implementation of the document-scanner pipeline.
// Each function maps to a single logical step.

// Step 1 – detectDocumentContour
std::vector<cv::Point> detectDocumentContour(const cv::Mat& image, bool debug,
                                              const std::string& outputDir)
{
    // Grayscale: colour is irrelevant for edge detection.
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // All geometric parameters were tuned on ~1000 px short side.
    // Higher-resolution photos need proportionally larger kernels so the
    // physical smoothing radius stays constant. Canny thresholds are
    // gradient-based and do not need to scale.
    const int    refShortSide = 1000;
    const int    shortSide    = std::min(image.rows, image.cols);
    const double scaleFactor  = static_cast<double>(shortSide) / refShortSide;

    // Kernel sizes must be odd and never drop below the baseline value.
    auto toOddKernel = [](double v, int baseline) -> int {
        int k = std::max(baseline, static_cast<int>(std::round(v)));
        return (k % 2 == 0) ? k + 1 : k;
    };

    const int blurK   = toOddKernel(5.0 * scaleFactor, 5);   // 5 @ 1000px
    const int dilateK = toOddKernel(3.0 * scaleFactor, 3);   // 3 @ 1000px

    // Gaussian blur: suppresses high-frequency noise so Canny reacts only
    // to meaningful edges.
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(blurK, blurK), 0);

    // Canny: pixels above cannyHigh are definite edges; pixels between
    // cannyLow and cannyHigh are kept only if adjacent to a definite edge.
    // Works well when the document lies on a contrasting background.
    const double cannyLow  = 50.0;
    const double cannyHigh = 150.0;
    cv::Mat edges;
    cv::Canny(blurred, edges, cannyLow, cannyHigh);

    if (debug)
        cv::imwrite(outputDir + "/debug_edges.png", edges);

    // Dilation: closes pixel-gaps in the document outline that would
    // otherwise split one contour into fragments.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               cv::Size(dilateK, dilateK));
    cv::dilate(edges, edges, kernel);

    if (debug)
        cv::imwrite(outputDir + "/debug_dilate.png", edges);

    // RETR_EXTERNAL: retrieves only outermost contours; inner edges
    // (text, table lines) are ignored. CHAIN_APPROX_SIMPLE saves memory.
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        std::cerr << "[detectDocumentContour] No contours found.\n";
        return {};
    }

    // Keep the largest quadrilateral that covers at least 10% of the image.
    // epsilon = 2% of perimeter is a standard heuristic that reduces a
    // slightly rounded rectangle to 4 vertices.
    const double imageArea       = static_cast<double>(image.rows * image.cols);
    const double minAreaFraction = 0.10;

    std::vector<cv::Point> bestContour;
    double bestArea = 0.0;

    for (const auto& c : contours)
    {
        double area = cv::contourArea(c);
        if (area < imageArea * minAreaFraction)
            continue;

        double perimeter = cv::arcLength(c, /*closed=*/true);
        double epsilon   = 0.02 * perimeter;
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, epsilon, /*closed=*/true);

        if (approx.size() != 4)
            continue;

        if (area > bestArea)
        {
            bestArea    = area;
            bestContour = approx;
        }
    }

    if (bestContour.empty())
        std::cerr << "[detectDocumentContour] No quadrilateral found.\n";
    else if (debug)
        cv::imwrite(outputDir + "/debug_contour.png", drawContourDebug(image, bestContour));

    return bestContour;
}


// Step 2 – orderCorners
std::vector<cv::Point2f> orderCorners(const std::vector<cv::Point>& corners)
{
    if (corners.size() != 4)
    {
        std::cerr << "[orderCorners] Expected 4 corners, got " << corners.size() << "\n";
        return {};
    }

    std::vector<cv::Point2f> ordered(4);

    // (x+y) and (y-x) uniquely identify each corner of a convex quadrilateral.
    float maxSum = -1e9f, minSum = 1e9f;
    float maxDiff = -1e9f, minDiff = 1e9f;
    cv::Point2f ptMaxSum, ptMinSum, ptMaxDiff, ptMinDiff;

    for (const auto& pt : corners)
    {
        float sum  = static_cast<float>(pt.x + pt.y);
        float diff = static_cast<float>(pt.y - pt.x);

        if (sum > maxSum)  { maxSum  = sum;  ptMaxSum  = pt; }
        if (sum < minSum)  { minSum  = sum;  ptMinSum  = pt; }
        if (diff > maxDiff){ maxDiff = diff; ptMaxDiff = pt; }
        if (diff < minDiff){ minDiff = diff; ptMinDiff = pt; }
    }

    // Order expected by getPerspectiveTransform: TL, TR, BR, BL.
    ordered[0] = ptMinSum;   // top-left     (smallest x+y)
    ordered[1] = ptMinDiff;  // top-right    (smallest y-x)
    ordered[2] = ptMaxSum;   // bottom-right (largest  x+y)
    ordered[3] = ptMaxDiff;  // bottom-left  (largest  y-x)

    return ordered;
}


// Step 3 – warpDocument
cv::Mat warpDocument(const cv::Mat& image,
                     const std::vector<cv::Point2f>& orderedCorners)
{
    const cv::Point2f& tl = orderedCorners[0];
    const cv::Point2f& tr = orderedCorners[1];
    const cv::Point2f& br = orderedCorners[2];
    const cv::Point2f& bl = orderedCorners[3];

    // Output size = max of the two parallel edges, so no data is lost
    // when the document is slightly trapezoidal.
    double widthTop    = cv::norm(tr - tl);
    double widthBottom = cv::norm(br - bl);
    int outWidth       = static_cast<int>(std::max(widthTop, widthBottom));

    double heightLeft  = cv::norm(bl - tl);
    double heightRight = cv::norm(br - tr);
    int outHeight      = static_cast<int>(std::max(heightLeft, heightRight));

    std::vector<cv::Point2f> dst = {
        {0.0f,                  0.0f},
        {(float)(outWidth - 1), 0.0f},
        {(float)(outWidth - 1), (float)(outHeight - 1)},
        {0.0f,                  (float)(outHeight - 1)}
    };

    // Compute the 3x3 homography and apply it.
    cv::Mat H = cv::getPerspectiveTransform(orderedCorners, dst);
    cv::Mat warped;
    cv::warpPerspective(image, warped, H, cv::Size(outWidth, outHeight),
                        cv::INTER_LINEAR);

    return warped;
}


// Step 4 – binarizeDocument (pre-blur + adaptive threshold)
cv::Mat binarizeDocument(const cv::Mat& warpedImage)
{
    cv::Mat gray;
    cv::cvtColor(warpedImage, gray, cv::COLOR_BGR2GRAY);

    // Pre-blur to suppress JPEG artefacts and paper grain (1-3 px signals).
    // Coefficient 2.0 and explicit sigma 0.6 keep smoothing light enough
    // not to spread ink strokes into the surrounding background.
    const int refWidth  = 800;
    const double bScale = static_cast<double>(warpedImage.cols) / refWidth;
    auto toOddKernel = [](double v, int baseline) -> int {
        int k = std::max(baseline, static_cast<int>(std::round(v)));
        return (k % 2 == 0) ? k + 1 : k;
    };
    const int blurK = toOddKernel(2.0 * bScale, 3);   // 3 px @ 800 px wide
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(blurK, blurK), 0.6);

    // Adaptive threshold instead of Otsu: phone photos have uneven lighting
    // (shadows, reflections) that a single global threshold handles poorly.
    // The local Gaussian mean adapts per-pixel so both bright and dark zones
    // are binarized correctly.
    // blockSize scales with resolution (baseline 41 px @ 800 px wide).
    // C=11 (raised from default 7): a pixel must be more distinctly darker
    // than its local mean to count as ink, rejecting noise that survives
    // the pre-blur.
    const int blockSize = toOddKernel(41.0 * bScale, 41);
    const int C         = 11;

    cv::Mat binary;
    cv::adaptiveThreshold(blurred,
                          binary,
                          255,                            // max value for white pixels
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, // weight neighbours by distance
                          cv::THRESH_BINARY,              // foreground < threshold → 0 (black)
                          blockSize,                      // neighbourhood scales with resolution
                          C);                             // stricter than default 7

    return binary;
}


// Step 5 – computeHorizontalProjection
std::vector<int> computeHorizontalProjection(const cv::Mat& binaryImage)
{
    // THRESH_BINARY: text = BLACK (0), background = WHITE (255).
    // Count black pixels per row: peaks = text lines, valleys = gaps.
    const int rows = binaryImage.rows;
    const int cols = binaryImage.cols;

    std::vector<int> profile(rows, 0);

    for (int r = 0; r < rows; ++r)
    {
        const uchar* rowPtr = binaryImage.ptr<uchar>(r);
        for (int c = 0; c < cols; ++c)
        {
            if (rowPtr[c] == 0) // black pixel = ink
                profile[r]++;
        }
    }

    return profile;
}


// Step 6 – segmentTextLines
std::vector<cv::Mat> segmentTextLines(const cv::Mat& binaryImage,
                                       const std::vector<int>& projection,
                                       int minLineHeight,
                                       bool debug,
                                       const std::string& outputDir)
{
    // Scan the projection for gap→text and text→gap transitions.
    // A row is "text" if its black-pixel count exceeds inkThreshold.

    // 1% of width: filters noise specks and thin ruling lines.
    const int inkThreshold = std::max(1, binaryImage.cols / 100);

    // rows/150 found experimentally: rows/100 discarded real short lines
    // at high resolution because their detected height fell below the threshold.
    const int effectiveMinHeight = std::max(minLineHeight, binaryImage.rows / 150);

    std::vector<cv::Mat> lines;

    bool inTextRegion = false;
    int regionStart   = 0;

    for (int r = 0; r < static_cast<int>(projection.size()); ++r)
    {
        bool rowHasInk = (projection[r] > inkThreshold);

        if (!inTextRegion && rowHasInk)
        {
            // Transition: gap → text.
            inTextRegion = true;
            regionStart  = r;
        }
        else if (inTextRegion && !rowHasInk)
        {
            // Transition: text → gap.
            int regionEnd = r - 1;
            int height    = regionEnd - regionStart + 1;

            if (height >= effectiveMinHeight)
            {
                cv::Rect roi(0, regionStart, binaryImage.cols, height);
                lines.push_back(binaryImage(roi).clone());
            }

            inTextRegion = false;
        }
    }

    // Handle the last region if it reaches the bottom of the image.
    if (inTextRegion)
    {
        int regionEnd = static_cast<int>(projection.size()) - 1;
        int height    = regionEnd - regionStart + 1;
        if (height >= effectiveMinHeight)
        {
            cv::Rect roi(0, regionStart, binaryImage.cols, height);
            lines.push_back(binaryImage(roi).clone());
        }
    }

    // Debug: save the projection as a bar chart (black bars, white gaps).
    if (debug && !projection.empty())
    {
        // Get the maximum value by deferencing the pointer the max element (iterator)
        int maxVal = *std::max_element(projection.begin(), projection.end());

        const int canvasWidth = 400;

        // Create a white (255) filling into a grayscale (CV_8UC1) matrix
        cv::Mat profileCanvas(static_cast<int>(projection.size()),
                              canvasWidth, CV_8UC1, cv::Scalar(255));

        if (maxVal > 0)
        {
            for (int r = 0; r < static_cast<int>(projection.size()); ++r)
            {
                int barLen = static_cast<int>((projection[r] / static_cast<double>(maxVal))
                                              * (canvasWidth - 1));
                cv::line(profileCanvas, cv::Point(0, r),
                         cv::Point(barLen, r), cv::Scalar(0), 1);
            }
        }

        cv::imwrite(outputDir + "/debug_projection.png", profileCanvas);
    }

    return lines;
}
