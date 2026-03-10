# Document Scanner — A1

C++ / OpenCV pipeline that takes a photo of a document on a table, corrects the perspective and segments the text lines.

## Build

Requirements: CMake ≥ 3.10, OpenCV 4.x.

```bash
mkdir build && cd build
cmake ..
make
```

## Run

```bash
./document_scanner ../test_image_1.jpg
./document_scanner ../test_image_1.jpg --debug
```

Output is saved to `build/output/<image_stem>/`. 
The `--debug` flag saves intermediate images (Canny edge map, detected contour, projection profile) to the same folder, useful for diagnosing failures.

## Output files

| File | Content |
|---|---|
| `corrected_document.png` | Perspective-corrected document |
| `binarized_document.png` | Black-and-white version |
| `line_1.png`, `line_2.png`, … | Individual text line crops |
