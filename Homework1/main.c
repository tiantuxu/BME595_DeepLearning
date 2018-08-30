#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

int c_conv(int in_channel, int o_channel, int kernel_size, int stride) {
    float ***kernel = (float***)malloc(in_channel*sizeof(float**));
    float ***out_array = (float***)malloc(o_channel*sizeof(float***));
    
    int Num_of_ops = 0;

    int i, j, k;
    int c1, c2, c3;

    static int rows = 720;
    static int columns = 1280;
    //static int rows = 1080;
    //static int columns = 1920;

    float ***input_image;

    input_image = (float***)malloc(in_channel*sizeof(float**));
    for (i = 0; i < in_channel; i++)
        input_image[i] = (float**)malloc(rows*sizeof(float*));

    for (i = 0; i < in_channel; i++)
        for (j = 0; j < rows; j++)
            input_image[i][j] = (float*)malloc(columns*sizeof(float));

    // Create input test image
    for(i = 0; i < in_channel; i++){
        for(j = 0; j < rows; j++){
            for(k = 0; k < columns; k++){
                input_image[i][j][k] = rand()%255;
            }
        }
    }

    // Initialize a 3D kernel
    for (i = 0; i < in_channel; i++){
        kernel[i] = (float**)malloc(kernel_size*sizeof(float*));
    }


    for (i = 0; i < in_channel; i++){
        for (j = 0; j < kernel_size; j++){
            kernel[i][j] = (float*)malloc(kernel_size*sizeof(float));
        }
    }

    for(i = 0; i < in_channel; i++){
        for(j = 0; j < kernel_size; j++){
            for(k = 0; k < kernel_size; k++){
                kernel[i][j][k] = (float)(rand()%100-50.0)/50.0;
            }
        }
    }

    int out_rows = (int)((rows - kernel_size)/stride + 1);
    int out_columns = (int)((columns - kernel_size)/stride + 1);

    // Initialize a output tenor
    for (i = 0; i < o_channel; i++){
        out_array[i] = (float**)malloc(out_rows*sizeof(float*));
    }

    
    for (i = 0; i < o_channel; i++){
        for (j = 0; j < out_rows; j++){
            out_array[i][j] = (float*)malloc(out_columns*sizeof(float));
        }
    }

    for(i = 0; i < o_channel; i++)
        for(j = 0; j < out_rows; j++)
            for(k = 0; k < out_columns; k++)
                out_array[i][j][k] = 0;

    // Convolutions
    for(i = 0; i < o_channel; i++){
        for(j = 0; j < out_rows; j++){
            for(k = 0; k < out_columns; k++){
                int sum = 0;
                for(c1 = 0; c1 < in_channel; c1++){
                    for(c2 = 0; c2 < kernel_size; c2++){
                        for(c3 = 0; c3 < kernel_size; c3++){
                            sum += kernel[c1][c2][c3] * input_image[c1][j+c2][k+c3];
                            Num_of_ops += 2;
                        }
                    }
                }
                out_array[i][j][k] = sum;
            }
        }
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

    for(int i = 0; i < 11; i++) {
        start = clock();
        Num_of_ops = c_conv(in_channel, pow(2,i), kernel_size, stride);
        end = clock();
        total_time[i] = (double)(end - start) / CLOCKS_PER_SEC;
        printf("For i = %d, computation_time = %lf \n", i, total_time[i]);
    }

    return 0;
}
