#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"
#include "kernels.h"
#define BLOCKSIZE 256






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
