#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

int rows[2] = {1080, 720};
int columns[2] = {1920, 1280};
int image = 0;

int c_conv(in_channel, o_channel, kernel_size, stride, float ***input_image) {
    float ***kernel = (float***)malloc(in_channel*sizeof(float**));
    float ***out_array = (float***)malloc(o_channel*sizeof(float***));
    int Num_of_ops = 0;

    int i, j, k;
    int c1, c2, c3;

    // Initialize a 3D kernel
    for (i = 0; i < in_channel; i++){
        kernel[i] = (float**)malloc(kernel_size*sizeof(float*));
        for (j = 0; j < kernel_size; j++){
            kernel[i][j] = (float*)malloc(kernel_size*sizeof(float));
        }
    }

    for(i = 0; i < kernel_size; i++){
        for(j = 0; j < kernel_size; j++){
            for(k = 0; k < kernel_size; k++){
                kernel[i][j][k] = (float)(rand()%100-50.0)/50.0;
            }
        }
    }

    int rows = (rows[image] - kernel_size)/stride + 1;
    int columns = (columns[image] - kernel_size)/stride + 1;

    // Initialize a 3D image tensor
    for (i = 0; i < o_channel; i++){
        out_array[i] = (float**)malloc(rows*sizeof(float*));
        for (j = 0; j < rows; j++){
            out_array[i][j] = (float*)malloc(columns*sizeof(float));
        }
    }

    for(i = 0; i < o_channel; i++)
        for(j = 0; j < rows; j++)
            for(k = 0; k < columns; k++)
                out_img_array[i][j][k] = 0;

    // Convolutions
    for(i = 0; i < o_channel; i++)
        for(j = 0; j < rows; j++)
            for(k = 0; k < columns; k++){
                sum = 0;
                for(c1 = 0; c1 < kernel_size; c1++){
                    for(c2 = 0; c2 < kernel_size; c2++){
                        for(c3 = 0; c3 < kernel_size; c3++){
                            sum += kernel[c1][c2][c3] * input_image[i+c1][j+c2][k+c3];
                            Num_of_ops += 2;
                        }
                    }
                }
                out_img_array[i][j][k] = sum;
             }


    return Num_of_ops;
}

int main(){

    int in_channel = 3;
    float total_time[11];
    int stride = 1;
    int kernel_size = 3;
    int i, j, k;
    int Num_of_ops;

    clock_t start, end;
    float ***input_image[2];

    for(int img_count = 0; img_count < 2; img_count++){
        input_image[img_count] = (float***)malloc(in_channel*sizeof(float**));
        for (i = 0; i < in_channel; i++)
            input_image[i] = (float**)malloc(rows[img_count]*sizeof(float*));
            for (j = 0; j < rows[img_count]; j++)
                input_image[i][j] = (float*)malloc(columns[img_count]*sizeof(float));

        // Create input test image
        for(i = 0; i < in_channel; i++)
            for(j = 0; j < rows[img_count]; j++)
                for(k = 0; k < columns[img_count]; k++){
                    input_image[i][j][k] = rand()%255;
                }

        for(int i = 0; i < 11; i++) {
            start = clock();
            Num_of_ops = c_conv(in_channel, pow(2,i), kernel_size, stride, input_image[img_count]);   //Call c_conv function
            end = clock();
            total_time[j] = (double)(end - start) / CLOCKS_PER_SEC;
            printf("For image %d, i = %d, number_of_operations = %d, computation_time = %lf \n",  img_count, i, Num_of_ops, total_time[i]);
        }
        image += 1;
    }
}
