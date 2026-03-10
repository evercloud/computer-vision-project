#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "scanner.h"
#include "utils.h"

// Entry point. Ties the pipeline together and saves the output images.
// Usage: ./document_scanner <image_path> [--debug]

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path> [--debug]\n";
        return 1;
    }

    const std::string imagePath = argv[1];
    bool debug = false;
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--debug")
            debug = true;

    // Output goes to build/output/<stem>/, next to the executable.
    namespace fs = std::filesystem;
    const fs::path inputPath(imagePath);
    const fs::path execDir   = fs::canonical(fs::path(argv[0])).parent_path();
    const fs::path outputDir = execDir / "output" / inputPath.stem();
    if (fs::exists(outputDir))
        fs::remove_all(outputDir);
    fs::create_directories(outputDir);
    std::cout << "Output dir: " << outputDir << "\n";

    auto out = [&](const std::string& filename) -> std::string {
        return (outputDir / filename).string();
    };

    // 1. Load image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        std::cerr << "[error] Cannot open image: " << imagePath << "\n";
        return 1;
    }
    std::cout << "Loaded: " << imagePath
              << "  (" << image.cols << " x " << image.rows << " px)\n";

    // 2. Detect document contour
    //    Returns the 4 corners of the largest quadrilateral.
    std::vector<cv::Point> corners = detectDocumentContour(image, debug, outputDir.string());
    if (corners.empty())
    {
        std::cerr << "[error] Could not find a document in the image.\n"
                  << "Tip: make sure the document lies on a contrasting background.\n";
        return 1;
    }
    std::cout << "Document contour found.\n";

    // 3. Order corners and warp perspective
    std::vector<cv::Point2f> orderedCorners = orderCorners(corners);
    if (orderedCorners.empty())
    {
        std::cerr << "[error] Failed to order document corners.\n";
        return 1;
    }
    cv::Mat warped = warpDocument(image, orderedCorners);
    if (debug)
        cv::imwrite(out("debug_warped.png"), warped);
    saveImage(out("corrected_document.png"), warped);

    // 4. Binarize
    cv::Mat binary = binarizeDocument(warped);
    saveImage(out("binarized_document.png"), binary);

    // 5. Layout analysis – horizontal projection + line segmentation
    std::vector<int> projection = computeHorizontalProjection(binary);
    std::vector<cv::Mat> textLines = segmentTextLines(binary, projection,
                                                       /*minLineHeight=*/10,
                                                       debug,
                                                       outputDir.string());

    std::cout << "Text lines detected: " << textLines.size() << "\n";

    for (int i = 0; i < static_cast<int>(textLines.size()); ++i)
        saveImage(out("line_" + std::to_string(i + 1) + ".png"), textLines[i]);

    std::cout << "Done.\n";
    return 0;
}
