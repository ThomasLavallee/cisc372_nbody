__global__ void initializeAccelerationMatrix(vector3 **accels, vector3 *values);
__global__ void computeAccelerations(vector3 **accels, vector3 *hPos, double *mass);
__global__ void sumAccelerationRow(vector3 **accels, vector3 *hVel, vector3 *hPos);
