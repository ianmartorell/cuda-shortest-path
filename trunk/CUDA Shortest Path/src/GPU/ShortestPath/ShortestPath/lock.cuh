#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct Lock
{
    int mutex;

    __device__ void lock()
    {
        while(atomicCAS(&mutex, 0, 1) != 0);
    }

    __device__ void unlock()
    {
        atomicExch(&mutex, 0);
    }
};
