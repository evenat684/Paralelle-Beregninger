#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

void negative(uchar* image);
void interpolate(uchar* image, uchar* new);

int main() {
	uchar *image = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
	readbmp("before.bmp", image);
	// Alter the image here
	
	//Change color to negative
	negative(image);

	//Double size
	uchar *new = calloc(2 * XSIZE * 2 * YSIZE * 3, 1);
	interpolate(image, new);
	
	
	savebmp("after.bmp", new, 2*XSIZE, 2*YSIZE);
	free(image);
	
	return 0;
}

void negative(uchar* image){
	//Turn every color to the negative of the color by taking 255 - currrent value
	for (int i = 0; i<YSIZE; i++){
		for (int j = 0; j<XSIZE*3; j+=3){
			image[3 * i * XSIZE + j + 0] = 255 - image[3 * i * XSIZE + j + 0];
			image[3 * i * XSIZE + j + 1] = 255 - image[3 * i * XSIZE + j + 1];
			image[3 * i * XSIZE + j + 2] = 255 - image[3 * i * XSIZE + j + 2];
		}
	}
}


void interpolate(uchar* image, uchar*new){
	//Duplicate pixels
	for (int i = 0; i<YSIZE; i++){
		for (int j= 0; j<XSIZE*3; j+=3){
			new[3 * i * 4 * XSIZE + 2 * j + 0] = image[3 * i * XSIZE + j + 0];
			new[3 * i * 4 * XSIZE + 2 * j + 1] = image[3 * i * XSIZE + j + 1];
			new[3 * i * 4 * XSIZE + 2 * j + 2] = image[3 * i * XSIZE + j + 2];
			new[3 * i * 4 * XSIZE + 2 * j + 3] = image[3 * i * XSIZE + j + 0];
			new[3 * i * 4 * XSIZE + 2 * j + 4] = image[3 * i * XSIZE + j + 1];
			new[3 * i * 4 * XSIZE + 2 * j + 5] = image[3 * i * XSIZE + j + 2];
			new[3 * (2*i+1) * 2 * XSIZE + 2 * j + 0] = image[3 * i * XSIZE + j + 0];
			new[3 * (2*i+1) * 2 * XSIZE + 2 * j + 1] = image[3 * i * XSIZE + j + 1];
			new[3 * (2*i+1) * 2 * XSIZE + 2 * j + 2] = image[3 * i * XSIZE + j + 2];
			new[3 * (2*i+1) * 2 * XSIZE + 2 * j + 3] = image[3 * i * XSIZE + j + 0];
			new[3 * (2*i+1) * 2 * XSIZE + 2 * j + 4] = image[3 * i * XSIZE + j + 1];
			new[3 * (2*i+1) * 2 * XSIZE + 2 * j + 5] = image[3 * i * XSIZE + j + 2];
		}
	}
}