#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "libs/bitmap.h"

// Convolutional Kernel Examples, each with dimension 3,
// gaussian kernel with dimension 5

int sobelYKernel[] = {-1, -2, -1,
                       0,  0,  0,
                       1,  2,  1};

int sobelXKernel[] = {-1, -0, 1,
                      -2,  0, 2,
                      -1,  0, 1};

int laplacian1Kernel[] = { -1,  -4,  -1,
                           -4,  20,  -4,
                           -1,  -4,  -1};

int laplacian2Kernel[] = { 0,  1,  0,
                           1, -4,  1,
                           0,  1,  0};

int laplacian3Kernel[] = { -1,  -1,  -1,
                           -1,   8,  -1,
                           -1,  -1,  -1};

int gaussianKernel[] = { 1,  4,  6,  4, 1,
                         4, 16, 24, 16, 4,
                         6, 24, 36, 24, 6,
                         4, 16, 24, 16, 4,
                         1,  4,  6,  4, 1 };

char* const kernelNames[]       = { "SobelY",     "SobelX",     "Laplacian 1",    "Laplacian 2",    "Laplacian 3",    "Gaussian"     };
int* const kernels[]            = { sobelYKernel, sobelXKernel, laplacian1Kernel, laplacian2Kernel, laplacian3Kernel, gaussianKernel };
unsigned int const kernelDims[] = { 3,            3,            3,                3,                3,                5              };
float const kernelFactors[]     = { 1.0,          1.0,          1.0,              1.0,              1.0,              1.0 / 256.0    };

int const maxKernelIndex = sizeof(kernelDims) / sizeof(unsigned int);
int my_rank;
int comm_sz;

// Helper function to swap bmpImageChannel pointers

void swapImage(bmpImage **one, bmpImage **two) {
  bmpImage *helper = *two;
  *two = *one;
  *one = helper;
}



//Get data into sub images
void getdatatoimage(bmpImage *image, bmpImage *sub_image){
  if (!my_rank){

  }
}
// Apply convolutional kernel on image data
void applyKernel(pixel **out, pixel **in, unsigned int width, unsigned int height, int *kernel, unsigned int kernelDim, float kernelFactor) {
  unsigned int const kernelCenter = (kernelDim / 2);
  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = 0; x < width; x++) {
      unsigned int ar = 0, ag = 0, ab = 0;
      for (unsigned int ky = 0; ky < kernelDim; ky++) {
        int nky = kernelDim - 1 - ky;
        for (unsigned int kx = 0; kx < kernelDim; kx++) {
          int nkx = kernelDim - 1 - kx;

          int yy = y + (ky - kernelCenter);
          int xx = x + (kx - kernelCenter);
          if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height) {
            ar += in[yy][xx].r * kernel[nky * kernelDim + nkx];
            ag += in[yy][xx].g * kernel[nky * kernelDim + nkx];
            ab += in[yy][xx].b * kernel[nky * kernelDim + nkx];
          }
        }
      }
      if (ar || ag || ab) {
        ar *= kernelFactor;
        ag *= kernelFactor;
        ab *= kernelFactor;
        out[y][x].r = (ar > 255) ? 255 : ar;
        out[y][x].g = (ag > 255) ? 255 : ag;
        out[y][x].b = (ab > 255) ? 255 : ab;
      } else {
        out[y][x].r = 0;
        out[y][x].g = 0;
        out[y][x].b = 0;
      }
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
    fprintf(out, "  -k, --kernel     <kernel>        kernel index (0<=x<=%u) (2)\n", maxKernelIndex -1);
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s before.bmp after.bmp -i 10000\n", exec);
}

int oldlinearmain(int argc, char **argv) {
  /*
    Parameter parsing, don't change this!
   */
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  unsigned int kernelIndex = 2;
  int ret = 0;

  static struct option const long_options[] =  {
      {"help",       no_argument,       0, 'h'},
      {"kernel",     required_argument, 0, 'k'},
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
        goto graceful_exit;
      case 'k':
        parse = strtol(optarg, &endptr, 10);
        if (endptr == optarg || parse < 0 || parse >= maxKernelIndex) {
          help(argv[0], c, optarg);
          goto error_exit;
        }
        kernelIndex = (unsigned int) parse;
        break;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg) {
          help(argv[0], c, optarg);
          goto error_exit;
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind+1)) {
    help(argv[0],' ',"Not enough arugments");
    goto error_exit;
  }

  unsigned int arglen = strlen(argv[optind]);
  input = calloc(arglen + 1, sizeof(char));
  strncpy(input, argv[optind], arglen);
  optind++;

  arglen = strlen(argv[optind]);
  output = calloc(arglen + 1, sizeof(char));
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
    goto error_exit;
  }

  if (loadBmpImage(image, input) != 0) {
    fprintf(stderr, "Could not load bmp image '%s'!\n", input);
    freeBmpImage(image);
    goto error_exit;
  }

  printf("Apply kernel '%s' on image with %u x %u pixels for %u iterations\n", kernelNames[kernelIndex], image->width, image->height, iterations);


  //start time measuremnt
  double start_time;
  start_time = MPI_Wtime();
  // Here we do the actual computation!
  // image->data is a 2-dimensional array of pixel which is accessed row first ([y][x])
  // each pixel is a struct of 3 unsigned char for the red, blue and green colour channel
  bmpImage *processImage = newBmpImage(image->width, image->height);
  for (unsigned int i = 0; i < iterations; i ++) {
    applyKernel(processImage->data,
                image->data,
                image->width,
                image->height,
                kernels[kernelIndex],
                kernelDims[kernelIndex],
                kernelFactors[kernelIndex]
                );
    swapImage(&processImage, &image);
  }
  freeBmpImage(processImage);

  // TODO: implement time measurement to here
  double spentTime = MPI_Wtime()- start_time;
  printf("Time spent: %.3f seconds\n", spentTime);
  
  MPI_Finalize();
  //Write the image back to disk
  if (saveBmpImage(image, output) != 0) {
    fprintf(stderr, "Could not save output to '%s'!\n", output);
    freeBmpImage(image);
    goto error_exit;
  };

graceful_exit:
  ret = 0;
error_exit:
  if (input)
    free(input);
  if (output)
    free(output);
  return ret;
};


//A function to load a buffer into an image
void writetoimage(bmpImage *image,pixel *buffer){
  for (int y = 0; y < image->height; y++){
    for (int x = 0; x < image->width; x++){
        image->data[y][x] = buffer[y*image->width + x];
    }
  }
}


//Function to exchange borders, set up after paper in problem set
void haloExchange(int halo_size, bmpImage *sub_image){
    int neighbours;
    //Figure out which ranks has which neighbours
    if(my_rank == 0 || my_rank == comm_sz-1){
      neighbours = 1;
    }else{
      neighbours = 2;
    }

    //Initialize variables for asyncrounous sends 
    //need one request and one status for each isend and irecv
    MPI_Request *request = calloc(sizeof(MPI_Request), neighbours * 2);
    MPI_Status *status = calloc(sizeof(MPI_Status), neighbours * 2);
    int buffersize = sub_image->width*halo_size*sizeof(pixel);
    int i = 0;

    //start receives of south
    if(my_rank != 0){
      MPI_Irecv(sub_image->data[0], buffersize,
                MPI_BYTE, my_rank-1, 0, MPI_COMM_WORLD,
                &request[i]);
    i+=1;
    }
    
    //start receives of north
    if(my_rank != comm_sz-1){
      MPI_Irecv(sub_image->data[sub_image->height - halo_size], 
                buffersize, MPI_BYTE, my_rank + 1, 0, MPI_COMM_WORLD,
                &request[i]);
    i+=1;
    }
    
    //north borders send
    if(my_rank != comm_sz-1){
      MPI_Isend(sub_image->data[sub_image->height - 2*halo_size],
              buffersize,
              MPI_BYTE, my_rank+1, 
              0, MPI_COMM_WORLD,
              &request[i]);
    i+=1;
    }
    //South border sends
    if(my_rank != 0){
      MPI_Isend(sub_image->data[halo_size],
                buffersize, MPI_BYTE, my_rank-1, 0, MPI_COMM_WORLD,
                &request[i]);
    i+=1;
    }
    
    //Wait for all sends and receives to complete
    MPI_Waitall(2 * neighbours, request, status);

    //free pointers
    free(status);
    free(request);
}

int main(int argc, char **argv) {
  //Initialize MPI
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, & comm_sz);

  if(comm_sz == 1){
    //If the comm size is just 1, run the old main
    oldlinearmain(argc,argv);
  }else{

  //Initializing variables
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  unsigned int kernelIndex = 2;
  int ret = 0;
  bmpImage *image;
  pixel *initbuffer;
  double start_time;
  unsigned int total_height;
  unsigned int total_width;
  processvariables *vars = calloc(1, sizeof(processvariables));
  
  /*
    Parameter parsing, don't change this!
   */
  if(!my_rank){
  static struct option const long_options[] =  {
      {"help",       no_argument,       0, 'h'},
      {"kernel",     required_argument, 0, 'k'},
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
        goto graceful_exit;
      case 'k':
        parse = strtol(optarg, &endptr, 10);
        if (endptr == optarg || parse < 0 || parse >= maxKernelIndex) {
          help(argv[0], c, optarg);
          goto error_exit;
        }
        kernelIndex = (unsigned int) parse;
        break;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg) {
          help(argv[0], c, optarg);
          goto error_exit;
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind+1)) {
    help(argv[0],' ',"Not enough arugments");
    goto error_exit;
  }

  unsigned int arglen = strlen(argv[optind]);
  input = calloc(arglen + 1, sizeof(char));
  strncpy(input, argv[optind], arglen);
  optind++;

  arglen = strlen(argv[optind]);
  output = calloc(arglen + 1, sizeof(char));
  strncpy(output, argv[optind], arglen);
  optind++;

  /*
    End of Parameter parsing!
   */


  /*
    Create the BMP image and load it from disk.
   */
  image = newBmpImage(0,0);
  if (image == NULL) {
    fprintf(stderr, "Could not allocate new image!\n");
    goto error_exit;
  }

  if (loadBmpImage(image, input) != 0) {
    fprintf(stderr, "Could not load bmp image '%s'!\n", input);
    freeBmpImage(image);
    goto error_exit;
  }
  printf("Apply kernel '%s' on image with %u x %u pixels for %u iterations\n", kernelNames[kernelIndex], image->width, image->height, iterations);
  
  
  //starting time measurement
  start_time = MPI_Wtime();

  //assign image->rawdata to a buffer
  initbuffer = calloc(image->width*image->height, sizeof(pixel));
  initbuffer = image->rawdata;

  //set vars for scattering them later
  vars->height = image->height;
  vars->width = image->width;
  vars->iterations = iterations;
  vars->kernelIndex = kernelIndex;
  }
  //Broadcast variables
  MPI_Bcast(vars, sizeof(processvariables), MPI_BYTE, 0,  MPI_COMM_WORLD);
  
  //Assign all values from struct variables(not necessary, but convient)
  total_width = vars->width;
  total_height = vars->height;
  iterations = vars->iterations;
  kernelIndex = vars->kernelIndex;
  
  
  //get the heigth for each sub_process 
  int sub_height = total_height / comm_sz;
  int remaining_rows = total_height % comm_sz;
  int *heightvector = calloc(comm_sz,sizeof(int));

  //Variables for scatterv
  int *sends = calloc(comm_sz, sizeof(int));
  int *displs = calloc(comm_sz,sizeof(int));
  int sizeofsends = sub_height*total_width*sizeof(pixel);
  int halo_size = kernelDims[kernelIndex] / 2;
  pixel *sub_rawdata = calloc(total_width*(total_height + 2 * halo_size), sizeof(pixel));
  int sendvar = 0;
  
  
  //set heights, size and dispacements for scatterv
  for (int i = 0; i<comm_sz; i++){
    heightvector[i] = sub_height;
    sends[i] = sizeofsends;
    if (remaining_rows > 0){
      heightvector[i] += 1;
      sends[i] += total_width*sizeof(pixel);
      remaining_rows -= 1;
    }
    displs[i] = sendvar;
    sendvar += sends[i];
  }
  
  //Initialize sub_images
  int sub_size = total_width*heightvector[my_rank]*sizeof(pixel);
  bmpImage *sub_image = newBmpImage(total_width, heightvector[my_rank] + 2 * halo_size);
  bmpImage *process_sub_image = newBmpImage(total_width, heightvector[my_rank] + 2 * halo_size);
  //Scatter the data and assign it to sub_image
  MPI_Scatterv(initbuffer, sends,
              displs, MPI_BYTE, &sub_rawdata[halo_size*total_width],
              sends[0], MPI_BYTE, 0, MPI_COMM_WORLD);
  writetoimage(sub_image, sub_rawdata);

  //Apply the kernel
  for (unsigned int i = 0; i < iterations; i ++) {
    haloExchange(halo_size, sub_image);
    applyKernel(process_sub_image->data,
                sub_image->data,
                sub_image->width,
                sub_image->height,
                kernels[kernelIndex],
                kernelDims[kernelIndex],
                kernelFactors[kernelIndex]
                );
    swapImage(&process_sub_image, &sub_image);
  }
  freeBmpImage(process_sub_image);


  //Allocate buffer to receive data from ranks
  pixel *imagebuffer = calloc(total_height*total_width, sizeof(pixel));
  
  //gather the images
  MPI_Gatherv(sub_image->data[halo_size],
              sizeof(pixel)*sub_image->width*heightvector[my_rank],
              MPI_BYTE,
              imagebuffer, sends, displs,
              MPI_BYTE, 0,
              MPI_COMM_WORLD
              );
  
  //Overwrite original image
  if(!my_rank){
    writetoimage(image, imagebuffer);
  }
  //Calculate spent time
  if(!my_rank){
    double spentTime = MPI_Wtime() - start_time;
    printf("Time spent: %.3f seconds\n", spentTime);
  }
  freeBmpImage(sub_image);
  free(sub_rawdata);
  free(displs);
  free(sends);
  MPI_Finalize();

  if(my_rank == 0){
  //Write the image back to disk
  if (saveBmpImage(image, output) != 0) {
    fprintf(stderr, "Could not save output to '%s'!\n", output);
    freeBmpImage(image);
    goto error_exit;
  }
  };

graceful_exit:
  free(initbuffer);
  ret = 0;
error_exit:
  if (input)
    free(input);
  if (output)
    free(output);
  return ret;
  }
};