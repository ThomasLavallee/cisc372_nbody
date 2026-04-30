#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"
#define BLOCKSIZE 256


/* Initializes the acceleration matrix for a given row

Arguments:
vector3 **accels, the acceleration matrix
vector3 *values, the values array

Returns: None
Side Effects: modifies accels matrix
*/
__global__ void initializeAccelerationMatrix(vector3 **accels, vector3 *values) {
        // Calculate which row we are initializing
        int index = threadIdx.x + (blockIdx.x * blockDim.x);

        // Only initialize row if it is in bounds of the matrix
        if (index >= 0 && index < NUMENTITIES) {
                accels[index] = &values[index * NUMENTITIES];
        }
}


/* Computes the acceleration for a cell in the accels matrix

Arguments:
vector3 **accels, the acceleration matrix
vector3 *hPos, the position array

Returns: None
Side Effects: Modifies the accels matrix in one cell
*/
__global__ void computeAccelerations(vector3 **accels, vector3 *hPos, double *mass) {
        // Calculate the row and column for the cell we are calculatin
        int i = (blockIdx.x * blockDim.x) + threadIdx.x;
        int j = (blockIdx.y * blockDim.y) + threadIdx.y;


        // Only calculate cell if index is valid
        if ((i >= 0) && (i < NUMENTITIES) && (j >= 0) && (j < NUMENTITIES)) {
                if (i == j) {
                        // Planet has 0 effect on its own acceleration
                        FILL_VECTOR(accels[i][j], 0, 0, 0);
                } else {
                        vector3 distance;
                        for (int k = 0; k < 3; k++) {
                                distance[k] = hPos[i][k] - hPos[j][k];
                        }

                        double magnitude_sq = distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2];
                        double magnitude = sqrt(magnitude_sq);

                        double accelmag = -1*GRAV_CONSTANT*mass[j]/magnitude_sq;
                        FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
                }
        }
}


/* Compute the sum of the accelerations for a given row

Arguments:
vector3 **accels, the acceleration matrix
vector3 *hVel, the array of velocities
vector3 *hPos, the array of positions

Returns: None
Side Effects: Modifies hVel and hPos arrays
*/
__global__ void sumAccelerationRow(vector3 **accels, vector3 *hVel, vector3 *hPos) {
        // Determine what row to sum
        int row = threadIdx.x + (blockIdx.x * blockDim.x);
        int k;

        // Only calculate the sum if row is within range of the table
        if (row >= 0 && row < NUMENTITIES) {
                vector3 accel_sum = {0, 0, 0};

                // Calculate the sum across the column
                for (int column = 0; column < NUMENTITIES; column++) {
                        for (k = 0; k < 3; k++) {
                                accel_sum[k] += accels[row][column][k];
                        }
                }


                // Update the hVels and hPost arrays
                for (k = 0; k < 3; k++) {
                        hVel[row][k] += accel_sum[k] * INTERVAL;
                        hPos[row][k] += hVel[row][k] * INTERVAL;

                }
        }
}


//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	// Create acceleration matrix of size NUMENTITIES x NUMENTITIES
	//vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	//vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);

	vector3 *d_values;
	vector3 **d_accels;
	cudaMalloc((void **)&d_values, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	cudaMalloc((void **)&d_accels, sizeof(vector3 *) * NUMENTITIES);	

	
	// Initialize acceleration matrix in parallel
	int numBlocks = (NUMENTITIES + BLOCKSIZE - 1) / BLOCKSIZE;
	initializeAccelerationMatrix<<<numBlocks, BLOCKSIZE>>>(d_accels, d_values);
	cudaDeviceSynchronize();

	// Compute acceleration matrix in parallel
	dim3 dimBlock(16, 16);
	dim3 dimGrid((NUMENTITIES + dimBlock.x - 1) / dimBlock.x, (NUMENTITIES + dimBlock.y - 1) / dimBlock.y);
	computeAccelerations<<<dimGrid, dimBlock>>>(d_accels, d_hPos, d_mass);
	cudaDeviceSynchronize();

	// Compute sum of acceleration matrixes row in parallel and update hVel and hPos
	sumAccelerationRow<<<numBlocks, BLOCKSIZE>>>(d_accels, d_hPos, d_hVel);
	cudaDeviceSynchronize();

	// Free the acceleration matrix
	cudaFree(d_values);
	cudaFree(d_accels);	
}
