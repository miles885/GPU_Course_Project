#include "ImageUtils.h"

#include <algorithm>
#include <cmath>
#include <iostream>

//NOTE: The RGB to HSV math was found at https://en.wikipedia.org/wiki/HSL_and_HSV
//      under the "Hue and Chroma", "Lightness", and "Saturation" sections

/**
 * Loads image data from the specified file
 *
 * @param fileName The file to load image data from
 * @param format   The image format
 * @param bitmap   The image data
 *
 * @return Flag denoting success or failure
 */
__host__
int32_t loadImage(const std::string & fileName, FREE_IMAGE_FORMAT & format, FIBITMAP ** bitmap)
{
    format = FreeImage_GetFileType(fileName.c_str());

    // Check to see if the format is unknown
    if(format == FIF_UNKNOWN)
    {
        format = FreeImage_GetFIFFromFilename(fileName.c_str());

        // Check to see if the format could not be determined
        if(format == FIF_UNKNOWN)
        {
            std::cerr << "Unknown file format. Exiting!" << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Attempt to load the image data
    if(FreeImage_FIFSupportsReading(format))
    {
        *bitmap = FreeImage_Load(format, fileName.c_str());

        // Check to see if the image was not loaded correctly
        if(*bitmap == 0)
        {
            std::cerr << "Unable to load image!" << std::endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

/**
 * Retrieve the image data info
 *
 * @param bitmap       The image data
 * @param imageWidth   The width of the image to write
 * @param imageHeight  The height of the image to write
 * @param bitsPerPixel The bits per pixel of the image to write
 *
 *  @return Flag denoting success or failure
 */
__host__
int32_t getImageInfo(FIBITMAP ** bitmap, uint32_t & imageWidth, uint32_t & imageHeight, uint32_t & bitsPerPixel)
{
    FREE_IMAGE_COLOR_TYPE colorType = FreeImage_GetColorType(*bitmap);

    if(colorType == FIC_MINISBLACK)
    {
        std::cout << "Loaded grayscale image" << std::endl;
    }
    else if(colorType == FIC_RGB)
    {
        std::cout << "Loaded RGB image" << std::endl;
    }
    else if(colorType == FIC_RGBALPHA)
    {
        std::cout << "Loaded RGBA image" << std::endl;
    }

    imageWidth = FreeImage_GetWidth(*bitmap);
    imageHeight = FreeImage_GetHeight(*bitmap);

    std::cout << "Image dimensions: " << imageWidth << " x " << imageHeight << std::endl;
    std::cout << "Pitch: " << FreeImage_GetPitch(*bitmap) << std::endl;

    bitsPerPixel = FreeImage_GetBPP(*bitmap);

    std::cout << "Bits per pixel: " << bitsPerPixel << std::endl << std::endl;

    return EXIT_SUCCESS;
}

/**
 * Loads the pixel data into a buffer
 *
 * @param bitmap       The image data
 * @param imageWidth   The width of the image to write
 * @param imageHeight  The height of the image to write
 * @param bitsPerPixel The bits per pixel of the image to write
 * @param pixelData    The channel-separated pixel data
 *
 * @return Flag denoting success or failure
 */
__host__
int32_t loadPixelData(FIBITMAP ** bitmap, uint32_t imageWidth, uint32_t imageHeight, uint32_t bitsPerPixel, BYTE * pixelData)
{
    // Set pixel data
    uint32_t imageSize = imageWidth * imageHeight;
    uint32_t bytesPerPixel = bitsPerPixel / 8;

    for(uint32_t y = 0; y < imageHeight; y++)
    {
        BYTE * bits = FreeImage_GetScanLine(*bitmap, y);

        for(uint32_t x = 0; x < imageWidth; x++)
        {
            // Check if 8 bits per pixel
            if(bitsPerPixel == 8)
            {
                pixelData[(y * imageWidth) + x] = bits[FI_RGBA_BLUE];
            }
            // Check if 24 or 32 bits per pixel
            else if(bitsPerPixel == 24 || bitsPerPixel == 32)
            {
                pixelData[(y * imageWidth) + x] = bits[FI_RGBA_RED];
                pixelData[(y * imageWidth) + imageSize + x] = bits[FI_RGBA_GREEN];
                pixelData[(y * imageWidth) + (imageSize * 2) + x] = bits[FI_RGBA_BLUE];
            }

            bits += bytesPerPixel;
        }
    }

    return EXIT_SUCCESS;
}

/**
 * Saves the image data to disk
 *
 * @param fileName     The file to write the image data to
 * @param format       The file format
 * @param imageWidth   The width of the image to write
 * @param imageHeight  The height of the image to write
 * @param bitsPerPixel The bits per pixel of the image to write
 * @param pixelData    The channel-separated pixel data
 *
 * @return Flag denoting success or failure
 */
__host__
int32_t saveImage(const char * fileName,
                  const FREE_IMAGE_FORMAT & format, 
                  const uint32_t imageWidth, 
                  const uint32_t imageHeight, 
                  const uint32_t bitsPerPixel, 
                  const BYTE * pixelData)
{
    FIBITMAP * bitmap = FreeImage_Allocate(imageWidth, imageHeight, 24);

    if(!bitmap)
    {
        std::cerr << "Failed to allocate image!" << std::endl;
        return EXIT_FAILURE;
    }

    // Set bitmap color data
    uint32_t imageSize = imageWidth * imageHeight;
    RGBQUAD color;

    for(uint32_t y = 0; y < imageHeight; y++)
    {
        for(uint32_t x = 0; x < imageWidth; x++)
        {
            // Check if 8 bits per pixel
            if(bitsPerPixel == 8)
            {
                color.rgbRed = pixelData[(y * imageWidth) + x];
                color.rgbGreen = pixelData[(y * imageWidth) + x];
                color.rgbBlue = pixelData[(y * imageWidth) + x];
            }
            // Check if 24 or 32 bits per pixel
            else if(bitsPerPixel == 24 || bitsPerPixel == 32)
            {
                color.rgbRed = pixelData[(y * imageWidth) + x];
                color.rgbGreen = pixelData[(y * imageWidth) + imageSize + x];
                color.rgbBlue = pixelData[(y * imageWidth) + (imageSize * 2) + x];
            }

            FreeImage_SetPixelColor(bitmap, x, y, &color);
        }
    }

    uint32_t status = FreeImage_Save(format, bitmap, fileName, 0);

    if(status != 1)
    {
        std::cerr << "Failed to save image!" << std::endl;
        return EXIT_FAILURE;
    }

    // Cleanup
    FreeImage_Unload(bitmap);

    return EXIT_SUCCESS;
}

/**
 * Convert channel separated RGB pixel data to grayscale pixel data
 *
 * @param imageWidth      The width of the image to write
 * @param imageHeight     The height of the image to write
 * @param bitsPerPixel    The bits per pixel of the image to write
 * @param rgbPixelData    The channel-separated pixel data
 * @param outputPixelData The single channel pixel data
 *
 * @return Flag denoting success or failure
 */
__host__
int32_t rgbToGray(uint32_t imageWidth, 
                  uint32_t imageHeight, 
                  uint32_t bitsPerPixel, 
                  const BYTE * pixelData, 
                  BYTE * outputPixelData)
{
    uint32_t imageSize = imageWidth * imageHeight;

    if(bitsPerPixel == 8)
    {
        memcpy(outputPixelData, pixelData, imageSize * sizeof(BYTE));
    }
    else
    {
        for(uint32_t y = 0; y < imageHeight; y++)
        {
            for(uint32_t x = 0; x < imageWidth; x++)
            {
                BYTE r = pixelData[(y * imageWidth) + x];
                BYTE g = pixelData[(y * imageWidth) + imageSize + x];
                BYTE b = pixelData[(y * imageWidth) + (imageSize * 2) + x];

                // Average the RGB intensities
                outputPixelData[(y * imageWidth) + x] = (r + g + b) / 3;
            }
        }
    }

    return EXIT_SUCCESS;
}

/**
 * Convert channel separated RGB pixel data to HSV pixel data
 *
 * @param imageWidth      The width of the image to write
 * @param imageHeight     The height of the image to write
 * @param bitsPerPixel    The bits per pixel of the image to write
 * @param pixelType       The type of pixel data to apply the filter to
 * @param rgbPixelData    The channel-separated pixel data
 * @param outputPixelData The single channel pixel data
 *
 * @return Flag denoting success or failure
 */
__host__
int32_t rgbToHSV(uint32_t imageWidth, 
                 uint32_t imageHeight, 
                 uint32_t bitsPerPixel, 
                 PixelType pixelType, 
                 const BYTE * pixelData, 
                 BYTE * outputPixelData)
{
    uint32_t imageSize = imageWidth * imageHeight;

    if(bitsPerPixel == 8)
    {
        memcpy(outputPixelData, pixelData, imageSize * sizeof(BYTE));
    }
    else
    {
        for(uint32_t y = 0; y < imageHeight; y++)
        {
            for(uint32_t x = 0; x < imageWidth; x++)
            {
                BYTE r = pixelData[(y * imageWidth) + x];
                BYTE g = pixelData[(y * imageWidth) + imageSize + x];
                BYTE b = pixelData[(y * imageWidth) + (imageSize * 2) + x];

                float floatR = r / 255.0f;
                float floatG = g / 255.0f;
                float floatB = b / 255.0f;

                float min = std::min(std::min(floatR, floatG), floatB);
                float max = std::max(std::max(floatR, floatG), floatB);
                float chroma = max - min;

                float hue = 0;
                float sat = 0;
                float value = max;

                if(chroma > 0)
                {
                    // Hue
                    if(max == floatR)
                    {
                        hue = fmod((floatG - floatB) / chroma, 6.0f);
                    }
                    else if(max == floatG)
                    {
                        hue = ((floatB - floatR) / chroma) + 2.0f;
                    }
                    else
                    {
                        hue = ((floatR - floatG) / chroma) + 4.0f;
                    }

                    hue *= 60.0f;

                    if(hue < 0)
                    {
                        hue += 360.0f;
                    }

                    // Saturation
                    sat = chroma / max;
                }

                // Set output to pixel type
                if(pixelType == HUE)
                {
                    outputPixelData[(y * imageWidth) + x] = (BYTE) (hue * 255.0f / 360.0f);
                }
                else if(pixelType == SATURATION)
                {
                    outputPixelData[(y * imageWidth) + x] = (BYTE) (sat * 255.0f);
                }
                else
                {
                    outputPixelData[(y * imageWidth) + x] = (BYTE) (value * 255.0f);
                }
            }
        }
    }

    return EXIT_SUCCESS;
}