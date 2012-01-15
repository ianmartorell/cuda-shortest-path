#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CUDATimer
{
private:
    cudaEvent_t start;
    cudaEvent_t end;

public:
    CUDATimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start,0);
    }

    ~CUDATimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    float milliseconds_elapsed()
    {
        float elapsed_time;
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, start, end);
        return elapsed_time;
    }

    float seconds_elapsed()
    {
        return milliseconds_elapsed() / 1000.0;
    }
};
