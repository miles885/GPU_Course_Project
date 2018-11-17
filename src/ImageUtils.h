#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "FreeImage.h"

__host__
int32_t loadImage(const std::string & fileName, FREE_IMAGE_FORMAT & format, FIBITMAP ** bitmap);

__host__
int32_t getImageInfo(FIBITMAP ** bitmap, uint32_t & imageWidth, uint32_t & imageHeight, uint32_t & bitsPerPixel);

__host__
int32_t loadPixelData(FIBITMAP ** bitmap, uint32_t imageWidth, uint32_t imageHeight, uint32_t bitsPerPixel, BYTE * pixelData);

__host__
int32_t saveImage(const char * fileName,
                  const FREE_IMAGE_FORMAT & format, 
                  const uint32_t imageWidth, 
                  const uint32_t imageHeight, 
                  const uint32_t bitsPerPixel, 
                  const BYTE * pixelData);

__host__
int32_t rgbToGray(uint32_t imageWidth, uint32_t imageHeight, uint32_t bitsPerPixel, const BYTE * pixelData, BYTE * grayPixelData);

#endif
