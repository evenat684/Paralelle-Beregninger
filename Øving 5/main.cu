#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>

extern "C" {
    #include "libs/bitmap.h"
}

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__);  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
       if (code != cudaSuccess)
       {
                 fprintf(stderr,"GPUassert: %s %s %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
                       if (abort) exit(code);
                          
       }
}

// Convolutional Filter Examples, each with dimension 3,
// gaussian filter with dimension 5

int sobelYFilter[] = {-1, -2, -1,
                       0,  0,  0,
                       1,  2,  1};

int sobelXFilter[] = {-1, -0, 1,
                      -2,  0, 2,
                      -1,  0, 1};

int laplacian1Filter[] = { -1,  -4,  -1,
                           -4,  20,  -4,
                           -1,  -4,  -1};

int laplacian2Filter[] = { 0,  1,  0,
                           1, -4,  1,
                           0,  1,  0};

int laplacian3Filter[] = { -1,  -1,  -1,
                           -1,   8,  -1,
                           -1,  -1,  -1};

int gaussianFilter[] = { 1,  4,  6,  4, 1,
                         4, 16, 24, 16, 4,
                         6, 24, 36, 24, 6,
                         4, 16, 24, 16, 4,
                         1,  4,  6,  4, 1 };

const char* filterNames[]       = { "SobelY",     "SobelX",     "Laplacian 1",    "Laplacian 2",    "Laplacian 3",    "Gaussian"     };
int* const filters[]            = { sobelYFilter, sobelXFilter, laplacian1Filter, laplacian2Filter, laplacian3Filter, gaussianFilter };
unsigned int const filterDims[] = { 3,            3,            3,                3,                3,                5              };
float const filterFactors[]     = { 1.0,          1.0,          1.0,              1.0,              1.0,              1.0 / 256.0    };

int const maxFilterIndex = sizeof(filterDims) / sizeof(unsigned int);

void cleanup(char** input, char** output) {
    if (*input)
        free(*input);
    if (*output)
        free(*output);
}

void graceful_exit(char** input, char** output) {
    cleanup(input, output);
    exit(0);
}

void error_exit(char** input, char** output) {
    cleanup(input, output);
    exit(1);
}

// Helper function to swap bmpImageChannel pointers

void swapImageRawdata(pixel **one, pixel **two) {
  pixel *helper = *two;
  *two = *one;
  *one = helper;
}

void swapImage(bmpImage **one, bmpImage **two) {
  bmpImage *helper = *two;
  *two = *one;
  *one = helper;
}

int divplussremainder(int a, int b){
  int c = a/b;
  if (a%b != 0){
    c += a%b;
  }
  return c;
}

__global__ void applyKernelFilter_shared(pixel *out, pixel *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) {
  unsigned int const filterCenter = (filterDim / 2);


  //Set coordinate of current pixe
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

  //Defining size of image(useful for checks later)
  int imagesize = width*height;

  //Allocate and assign memory for shared filter
  __shared__ int s_filter[25];
  if (threadIdx.x < filterDim*filterDim){
    s_filter[threadIdx.x] = filter[threadIdx.x];
  }
  
  //Allocate and assign memory for shared pixels
  extern __shared__ pixel s_pixels[];
  unsigned int h_x = threadIdx.x + filterCenter;
  unsigned int h_y = threadIdx.y + filterCenter;


  //Assign values to pixels inside block
  if (x < width && y < height && h_x >= filterCenter && h_x < blockDim.x + filterCenter && h_y >= filterCenter && h_y < blockDim.y + filterCenter){
    s_pixels[h_x + (blockDim.x + 2 * filterCenter) * h_y] = in[x + y*width];
  }

  //Set west border
  if (threadIdx.x == 0){
    for (int i = 0; i < filterCenter; i++){
      if(x + y*width - i < imagesize){
      s_pixels[h_y*(blockDim.x + 2 * filterCenter) + i] = in[x + y*width -i]; 
      }
    }
  }
  
  //Set east border
  if (threadIdx.x == blockDim.x -1){
    for (int i = 0; i < filterCenter; i++){
      if(x + y * width + i < imagesize){
        s_pixels[h_y* (blockDim.x + 2 * filterCenter) + blockDim.x + filterCenter + i] = in[x + y*width +i]; 
      }
    }
  }
  
  
  //set North border
  if (threadIdx.y == blockDim.y -1){
    for (int i = 0; i < filterCenter; i++){
      if(x + (y + i + 1) * width< imagesize){
        s_pixels[(h_y + i + 1) * (blockDim.x + 2 * filterCenter) + threadIdx.x] = in[x + (y + i + 1)*width];
      }
    }
  }
  
  //Set South Border
  if(threadIdx.y == 0){
    for (int i = 0; i < filterCenter; i++){
      if(x + (y - i - 1) * width< imagesize && x + (y - i - 1) * width > 0){
        s_pixels[(i) * (blockDim.x + 2 * filterCenter) + threadIdx.x] = in[x + (y -  filterCenter + i)*width];
      }
    }
  }

  //Set south west corner:
  for(int i = 0; i<filterCenter; i++){
    for (int j = 0; j<filterCenter; j++){
      if(blockDim.x*blockIdx.x + blockDim.y*blockIdx.y*width - filterCenter + j - filterCenter*width + i> 0 && blockDim.x*blockIdx.x + blockDim.y*blockIdx.y*width - filterCenter + j - filterCenter*width + i < imagesize){
        s_pixels[i*(blockDim.x + 2 * filterCenter) + j] = in[blockDim.x*blockIdx.x + blockDim.y*blockIdx.y*width - filterCenter + j - filterCenter*width + i];
      }
    }
  }

  //Set south east corner
  for(int i = 0; i<filterCenter; i++){
    for (int j = 0; j<filterCenter; j++){
      if(blockDim.x*blockIdx.x + blockDim.x + blockDim.y*blockIdx.y*width - filterCenter + j - filterCenter*width + i*width> 0 && blockDim.x*blockIdx.x + blockDim.x + blockDim.y*blockIdx.y*width - filterCenter + j - filterCenter*width + i*width < imagesize){
        s_pixels[i*(blockDim.x + 2 * filterCenter) + j + blockDim.x] = in[blockDim.x*blockIdx.x + blockDim.x + blockDim.y*blockIdx.y*width - filterCenter + j - filterCenter*width + i*width];
      }
    }
  }

  //Set north west corner
  for(int i = 0; i<filterCenter; i++){
    for (int j = 0; j<filterCenter; j++){
      if(blockDim.x*blockIdx.x + blockDim.y*blockIdx.y*width - filterCenter + j - filterCenter*width + i*width + blockDim.y*width > 0 && blockDim.x*blockIdx.x + blockDim.y*blockIdx.y*width - filterCenter + j - filterCenter*width + i*width + blockDim.y*width < imagesize){
        s_pixels[(i + blockDim.y)*(blockDim.x + 2 * filterCenter) + j] = in[blockDim.x*blockIdx.x + blockDim.y*blockIdx.y*width - filterCenter + j - filterCenter*width + i*width + blockDim.y*width];
      }
    }
  }

  //Set north east corner
  for(int i = 0; i<filterCenter; i++){
    for (int j = 0; j<filterCenter; j++){
      if(blockDim.x*blockIdx.x + blockDim.y*blockIdx.y*width - filterCenter + j - filterCenter*width + i + blockDim.y*width + blockDim.x > 0 && blockDim.x*blockIdx.x + blockDim.y*blockIdx.y*width - filterCenter + j - filterCenter*width + i + blockDim.y*width + blockDim.x < imagesize){
        s_pixels[(i + blockDim.y)*(blockDim.x + 2 * filterCenter) + j + blockDim.x] = in[blockDim.x*blockIdx.x + blockDim.y*blockIdx.y*width - filterCenter + j - filterCenter*width + i + blockDim.y*width + blockDim.x];
      }
    }
  }
  __syncthreads();
  
  //Iterate through the filter
  int ar = 0; int ag = 0; int ab = 0;
  for (unsigned int ky = 0; ky < filterDim; ky++) {
    int nky = filterDim - 1 - ky;
    for (unsigned int kx = 0; kx < filterDim; kx++) {
      int nkx = filterDim - 1 - kx;
      int yy = h_y + (ky - filterCenter);
      int xx = h_x + (kx - filterCenter);
      if (xx >= 0 && xx < blockDim.x + 4*filterCenter && yy >= 0 && yy < blockDim.y + 4*filterCenter){
        ar += s_pixels[xx + yy*(blockDim.x + filterCenter * 2)].r * s_filter[nky * filterDim + nkx];
        ag += s_pixels[xx + yy*(blockDim.x + filterCenter * 2)].g * s_filter[nky * filterDim + nkx];
        ab += s_pixels[xx + yy*(blockDim.x + filterCenter * 2)].b * s_filter[nky * filterDim + nkx];
      }
    }
  }
  ar *= filterFactor;
  ag *= filterFactor;
  ab *= filterFactor;
  
  ar = (ar < 0) ? 0 : ar;
  ag = (ag < 0) ? 0 : ag;
  ab = (ab < 0) ? 0 : ab;

  //Set values in out 
  if(x < width && y < height){
    out[y*width + x].r = (ar > 255) ? 255 : ar;
    out[y*width + x].g = (ag > 255) ? 255 : ag;
    out[y*width + x].b = (ab > 255) ? 255 : ab;
  }
}


__global__ void applyKernelFilter_paralell(pixel *out, pixel *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) {
  unsigned int const filterCenter = (filterDim / 2);
  //Set coordinate of current pixel
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  int ar = 0; int ag = 0; int ab = 0;
  for (unsigned int ky = 0; ky < filterDim; ky++) {
    int nky = filterDim - 1 - ky;
    for (unsigned int kx = 0; kx < filterDim; kx++) {
      int nkx = filterDim - 1 - kx;
      int yy = y + (ky - filterCenter);
      int xx = x + (kx - filterCenter);
      if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height){
        ar += in[yy*width + xx].r * filter[nky * filterDim + nkx];
        ag += in[yy*width + xx].g * filter[nky * filterDim + nkx];
        ab += in[yy*width + xx].b * filter[nky * filterDim + nkx];
      }
    }
  }
  ar *= filterFactor;
  ag *= filterFactor;
  ab *= filterFactor;
  
  ar = (ar < 0) ? 0 : ar;
  ag = (ag < 0) ? 0 : ag;
  ab = (ab < 0) ? 0 : ab;

  //Set values in out 
  if(x < width && y<height){
  out[y*width + x].r = (ar > 255) ? 255 : ar;
  out[y*width + x].g = (ag > 255) ? 255 : ag;
  out[y*width + x].b = (ab > 255) ? 255 : ab;
  }
}


// Apply convolutional filter on image data
__global__ void applyKernelFilter(pixel *out, pixel *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) {
  unsigned int const filterCenter = (filterDim / 2);
  /*
  //Set coordinate of current pixel
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  */
  for (unsigned int y = 0; y<height; y++){
    for (unsigned int x = 0; x<width; x++){
  //Iterate through the filter
  int ar = 0; int ag = 0; int ab = 0;
  for (unsigned int ky = 0; ky < filterDim; ky++) {
    int nky = filterDim - 1 - ky;
    for (unsigned int kx = 0; kx < filterDim; kx++) {
      int nkx = filterDim - 1 - kx;
      int yy = y + (ky - filterCenter);
      int xx = x + (kx - filterCenter);
      if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height){
        ar += in[yy*width + xx].r * filter[nky * filterDim + nkx];
        ag += in[yy*width + xx].g * filter[nky * filterDim + nkx];
        ab += in[yy*width + xx].b * filter[nky * filterDim + nkx];
      }
    }
  }
  ar *= filterFactor;
  ag *= filterFactor;
  ab *= filterFactor;
  
  ar = (ar < 0) ? 0 : ar;
  ag = (ag < 0) ? 0 : ag;
  ab = (ab < 0) ? 0 : ab;

  //Set values in out array
  out[y*width + x].r = (ar > 255) ? 255 : ar;
  out[y*width + x].g = (ag > 255) ? 255 : ag;
  out[y*width + x].b = (ab > 255) ? 255 : ab;
  }
}
}



void help(char const *exec, char const opt, char const *optarg) {
    FILE *out = stdout;
    if (opt != 0) {
        out = stderr;
        if (optarg) {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        } else {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }
    fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options:\n");
    fprintf(out, "  -k, --filter     <filter>        filter index (0<=x<=%u) (2)\n", maxFilterIndex -1);
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s before.bmp after.bmp -i 10000\n", exec);
}


int main(int argc, char **argv) {
  /*
    Parameter parsing, don't change this!
   */
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  unsigned int filterIndex = 2;

  static struct option const long_options[] =  {
      {"help",       no_argument,       0, 'h'},
      {"filter",     required_argument, 0, 'k'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}
  };

  static char const * short_options = "hk:i:";
  {
    char *endptr;
    int c;
    int parse;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
      switch (c) {
      case 'h':
        help(argv[0],0, NULL);
        graceful_exit(&input,&output);
      case 'k':
        parse = strtol(optarg, &endptr, 10);
        if (endptr == optarg || parse < 0 || parse >= maxFilterIndex) {
          help(argv[0], c, optarg);
          error_exit(&input,&output);
        }
        filterIndex = (unsigned int) parse;
        break;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg) {
          help(argv[0], c, optarg);
          error_exit(&input,&output);
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind+1)) {
    help(argv[0],' ',"Not enough arugments");
    error_exit(&input,&output);
  }

  unsigned int arglen = strlen(argv[optind]);
  input = (char*)calloc(arglen + 1, sizeof(char));
  strncpy(input, argv[optind], arglen);
  optind++;

  arglen = strlen(argv[optind]);
  output = (char*)calloc(arglen + 1, sizeof(char));
  strncpy(output, argv[optind], arglen);
  optind++;

  /*
    End of Parameter parsing!
   */


  /*
    Create the BMP image and load it from disk.
   */
  bmpImage *image = newBmpImage(0,0);
  if (image == NULL) {
    fprintf(stderr, "Could not allocate new image!\n");
    error_exit(&input,&output);
  }

  if (loadBmpImage(image, input) != 0) {
    fprintf(stderr, "Could not load bmp image '%s'!\n", input);
    freeBmpImage(image);
    error_exit(&input,&output);
  }

  printf("Apply filter '%s' on image with %u x %u pixels for %u iterations\n", filterNames[filterIndex], image->width, image->height, iterations);


  // TODO: implement time measurement from here
  struct timespec start_time, end_time;
  

  // TODO: Cuda malloc and memcpy the rawdata from the images and filter, from host side to device side.
  pixel *d_image;
  pixel *d_process;
  int *d_filter;
  int imagesize =  image->width*image->height*sizeof(pixel);
  int filtersize = filterDims[filterIndex]*filterDims[filterIndex]*sizeof(int);
  cudaMalloc((void**) &d_image, imagesize);
  cudaMalloc((void**) &d_process, imagesize);
  cudaMalloc((void**) &d_filter, filtersize);
  cudaMemcpy(d_image, image->rawdata, imagesize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_process, image->rawdata, imagesize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, filters[filterIndex], filtersize, cudaMemcpyHostToDevice);

  
  
  int blockSize;
  int minGridsize;


  cudaOccupancyMaxPotentialBlockSize(&minGridsize, &blockSize, applyKernelFilter_shared, 0 ,0);
  const dim3 gridsize(divplussremainder(image->width, blockSize/32), divplussremainder(image->height, blockSize/32)); 
  const dim3 threadblock(blockSize/32, blockSize/32);
  int size_of_halomem = (filterDims[filterIndex] / 2 ) * 2;
  int shared_mem_size = (threadblock.x + size_of_halomem)* (threadblock.y + size_of_halomem) * sizeof(pixel);
  
  
  //Start timer
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (unsigned int i = 0; i < iterations; i ++) {
    //Run paralell kernel if total number of threads exceeds number of pixels
    applyKernelFilter_shared<<<gridsize, threadblock, shared_mem_size>>>(d_process,
      d_image,
      image->width,
      image->height,
      d_filter,
      filterDims[filterIndex],
      filterFactors[filterIndex]
      );
    cudaDeviceSynchronize(); 
    swapImageRawdata(&d_process, &d_image);
  }

  // TODO: Copy back rawdata from images
  cudaMemcpy(image->rawdata, d_image, imagesize, cudaMemcpyDeviceToHost);
  
  // TODO: Stop CUDA timer
  clock_gettime(CLOCK_MONOTONIC, &end_time);


  //Theoretical occupancy
  int maxBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, applyKernelFilter_paralell, blockSize, 0);
  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);
  float occupancy = (maxBlocks * blockSize / props.warpSize) / (float)(props.maxThreadsPerMultiProcessor / props.warpSize);
  printf("Launched blocks of size %d. Theoretical occupancy: %f \n", blockSize, occupancy);

  // TODO: Calculate and print elapsed time
  float spentTime = ((double) (end_time.tv_sec - start_time.tv_sec)) + ((double) (end_time.tv_nsec - start_time.tv_nsec)) * 1e-9;
  printf("Time spent: %.3f seconds\n", spentTime);
  cudaFree(d_filter);
  cudaFree(d_image);
  cudaFree(d_process);
  //Write the image back to disk
  if (saveBmpImage(image, output) != 0) {
    fprintf(stderr, "Could not save output to '%s'!\n", output);
    freeBmpImage(image);
    error_exit(&input,&output);
  };

  graceful_exit(&input,&output);
};
