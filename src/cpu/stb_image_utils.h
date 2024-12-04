#ifndef STB_IMAGE_UTILS_H
#define STB_IMAGE_UTILS_H

// Disable strict warnings for this header from the Microsoft Visual C++ compiler.
#ifdef _MSC_VER
    #pragma warning (push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>

class rtw_image {
  public:
    rtw_image() {}

    rtw_image(const char* image_filename) {

        auto filename = std::string(image_filename);
        auto imagedir = getenv("RTW_IMAGES");

        // Hunt for the image file in some likely locations.
        if (imagedir && load(std::string(imagedir) + "/" + image_filename)) return;
        if (load(filename)) return;
        if (load("images/" + filename)) return;
        if (load("../images/" + filename)) return;
        if (load("../../images/" + filename)) return;
        if (load("../../../images/" + filename)) return;
        if (load("../../../../images/" + filename)) return;
        if (load("../../../../../images/" + filename)) return;
        if (load("../../../../../../images/" + filename)) return;

        std::cerr << "ERROR: Could not load image file '" << image_filename << "'.\n";
    }

    ~rtw_image() {
        delete[] bdata;
        STBI_FREE(fdata);
    }

    bool load(const std::string& filename) {

        auto n = bytes_per_pixel; // Dummy out parameter: original components per pixel
        fdata = stbi_loadf(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);
        if (fdata == nullptr) return false;

        bytes_per_scanline = image_width * bytes_per_pixel;
        convert_to_bytes();
        return true;
    }

    int width()  const { return (fdata == nullptr) ? 0 : image_width; }
    int height() const { return (fdata == nullptr) ? 0 : image_height; }

    const unsigned char* pixel_data(int x, int y) const {
        // Return the address of the three RGB bytes of the pixel at x,y. If there is no image
        // data, returns magenta.
        static unsigned char magenta[] = { 255, 0, 255 };
        if (bdata == nullptr) return magenta;

        x = clamp(x, 0, image_width);
        y = clamp(y, 0, image_height);

        return bdata + y*bytes_per_scanline + x*bytes_per_pixel;
    }

  private:
    const int      bytes_per_pixel = 3;
    float         *fdata = nullptr;         // Linear floating point pixel data
    unsigned char *bdata = nullptr;         // Linear 8-bit pixel data
    int            image_width = 0;         // Loaded image width
    int            image_height = 0;        // Loaded image height
    int            bytes_per_scanline = 0;

    static int clamp(int x, int low, int high) {
        // Return the value clamped to the range [low, high).
        if (x < low) return low;
        if (x < high) return x;
        return high - 1;
    }

    static unsigned char float_to_byte(float value) {
        if (value <= 0.0)
            return 0;
        if (1.0 <= value)
            return 255;
        return static_cast<unsigned char>(256.0 * value);
    }

    void convert_to_bytes() {
        // Convert the linear floating point pixel data to bytes, storing the resulting byte
        // data in the `bdata` member.

        int total_bytes = image_width * image_height * bytes_per_pixel;
        bdata = new unsigned char[total_bytes];

        // Iterate through all pixel components, converting from [0.0, 1.0] float values to
        // unsigned [0, 255] byte values.

        auto *bptr = bdata;
        auto *fptr = fdata;
        for (auto i=0; i < total_bytes; i++, fptr++, bptr++)
            *bptr = float_to_byte(*fptr);
    }
};

void parse_obj(const char* filename, 
                std::vector<vec3>& points, 
                std::vector<vec3>& idxVertex, 
                int& nPoints, 
                int& nTriangles) {

    std::ifstream objFile(filename); 
    if (!objFile.is_open()) {
        std::cerr << "Error: Cannot open the OBJ file: " << filename << std::endl;
        return;
    }

    int np = 0, nt = 0; // Initialize counters for vertices and triangles
    std::string line;

    while (std::getline(objFile, line)) {
        std::stringstream ss(line);
        std::string label;
        ss >> label;

        if (label == "v") {
            vec3 vertex;
            ss >> vertex[0] >> vertex[1] >> vertex[2];
            points[np++] = vertex;
        } else if (label == "f") {
            vec3 idx;
            ss >> idx[0] >> idx[1] >> idx[2];
            idx[0] -= 1; idx[1] -= 1; idx[2] -= 1; // Adjust to 0-based indexing
            idxVertex[nt++] = idx;
        }
    }

    objFile.close(); // Close the file

    // Update the number of vertices and triangles
    nPoints = np;
    nTriangles = nt;

    // Optional: Centering and scaling
    vec3 mean = vec3(0, 0, 0);
    for (int i = 0; i < nPoints; i++) {
        mean += points[i];
    }
    mean /= float(nPoints);

    for (int i = 0; i < nPoints; i++) {
        points[i] += -mean;
    }

    float maxDistance = 0.0f;
    for (int i = 0; i < nPoints; i++) {
        float dist = points[i].length();
        if (dist > maxDistance) {
            maxDistance = dist;
        }
    }

    for (int i = 0; i < nPoints; i++) {
        points[i] /= maxDistance;
    }
}                

// Restore MSVC compiler warnings
#ifdef _MSC_VER
    #pragma warning (pop)
#endif


#endif
