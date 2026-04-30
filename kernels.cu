#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

/** Initializes a row of the acceleration matrix 
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
