#include <stdint.h>
#include <stdio.h>
#include "config.h"

typedef struct Image_{
    uint8_t pixels[IMAGE_SIZE];
} Image;

typedef struct ImageFileHeader_{
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
} ImageFileHeader;

typedef struct LabelFileHeader_{
    uint32_t magicNumber;
    uint32_t maxLabels;
} LabelFileHeader;

void getImage(FILE *imageFile, Image* img);
uint8_t getLabel(FILE *labelFile);
FILE* openImageFile(char *fileName, ImageFileHeader* imageFileHeader);
FILE* openLabelFile(char *fileName);


//Image struct mei pura array image ka
//image Header mei width,height and no of images 
//label file header mei max label