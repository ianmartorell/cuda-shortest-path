#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "csr_bellman.cuh"
using namespace std;
using namespace thrust;

void test_csr_bellman(int source, int target, string filename, bool printPath, bool isScalar)
{
    cout << "Reading Graph..." << endl;

    ReadCSRGraph(filename);

    if (target == -1)
        target = g_size - 1;

    cout << "Doing ShortestPath Algorithm..." << endl;

    CUDATimer timer;

    // in normal case, if (edge_num / g_size < 32) scalar is more effective, else vector
    int dist = isScalar ? csr_bellman_scalar(source, target, g_size) : csr_bellman_vector(source, target, g_size);

    string method = isScalar ? "CUDA CSR Bellman (Scalar)" : "CUDA CSR Bellman (Vector)";

    PrintResult(method, source, target, dist, timer.seconds_elapsed(), printPath);

    free(g);
    free(gSize);
    free(gIndex);
    free(parent);
    cudaFree(dev_g);
    cudaFree(dev_gSize);
    cudaFree(dev_gIndex);
}

int csr_bellman_vector(int source, int target, int size)
{    
    // dist1 and dist2 are used as rolling array 
    device_vector<int> dist1(size, INT_MAX);
    device_vector<int> dist2(size, INT_MAX);

    dist1[source] = dist2[source] = 0;

    int *dev_dist1_ptr = raw_pointer_cast(&dist1[0]);
    int *dev_dist2_ptr = raw_pointer_cast(&dist2[0]);

    device_vector<int> dev_parent(size, -1);
    int *dev_parent_ptr = raw_pointer_cast(&dev_parent[0]);

    bool flag = true; // determine dist1 and dist2 is distS or distD
 
    // record whether the dist is changed
    device_vector<bool> change(size, false);
    bool *dev_change_ptr = raw_pointer_cast(&change[0]);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    const int threadsPerBlock = 1024;
    const int ThreadsPerVector = 32;
    const int VectorsPerBlock = threadsPerBlock / ThreadsPerVector;
    const int blocksPerGrid = std::min(prop.maxGridSize[0], (size + VectorsPerBlock - 1) / VectorsPerBlock);
    
    while (true)
    {
        if (flag)
        {
            relax_vector<VectorsPerBlock, ThreadsPerVector><<<blocksPerGrid, threadsPerBlock>>>
                (dev_dist2_ptr, dev_g, dev_gIndex, dev_gSize, dev_dist1_ptr, size, dev_change_ptr, dev_parent_ptr);
        }
        else
        {
            relax_vector<VectorsPerBlock, ThreadsPerVector><<<blocksPerGrid, threadsPerBlock>>>
                (dev_dist1_ptr, dev_g, dev_gIndex, dev_gSize, dev_dist2_ptr, size, dev_change_ptr, dev_parent_ptr);
        }

        flag = !flag;
        
        // if there is one dist changed, continue doing, else break
        bool isChanged = reduce(change.begin(), change.end(), false, thrust::logical_or<bool>());

        if (!isChanged)
            break;
    }

    cudaMemcpy(parent, dev_parent_ptr, sizeof(int) * size, cudaMemcpyDeviceToHost);

    int ret = dist1[target];

    return ret;
}

int csr_bellman_scalar(int source, int target, int size)
{
    device_vector<int> dist1(size, INT_MAX);
    device_vector<int> dist2(size, INT_MAX);

    dist1[source] = dist2[source] = 0;

    int *dev_dist1_ptr = raw_pointer_cast(&dist1[0]);
    int *dev_dist2_ptr = raw_pointer_cast(&dist2[0]);

    device_vector<int> dev_parent(size, -1);
    int *dev_parent_ptr = raw_pointer_cast(&dev_parent[0]);
    
    bool flag = true; 

    device_vector<bool> change(size, false);
    bool *dev_change_ptr = raw_pointer_cast(&change[0]);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    const int threadsPerBlock = prop.maxThreadsDim[0];
    const int blocksPerGrid = std::min(prop.maxGridSize[0], (size + threadsPerBlock - 1) / threadsPerBlock);        

    while (true)
    {
        if (flag)
        {           
            relax_scalar<<<blocksPerGrid, threadsPerBlock>>>(dev_dist2_ptr, dev_g, dev_gIndex, dev_gSize, dev_dist1_ptr, size, dev_change_ptr, dev_parent_ptr);
        }
        else
        {
            relax_scalar<<<blocksPerGrid, threadsPerBlock>>>(dev_dist1_ptr, dev_g, dev_gIndex, dev_gSize, dev_dist2_ptr, size, dev_change_ptr, dev_parent_ptr);
        }

        flag = !flag;
        
        bool isChanged = reduce(change.begin(), change.end(), false, thrust::logical_or<bool>());

        if (!isChanged)
            break;
    }    

    cudaMemcpy(parent, dev_parent_ptr, sizeof(int) * size, cudaMemcpyDeviceToHost);

    int ret = dist1[target];

    return ret;
}

__global__ void relax_scalar(int *distD, int *g, int *gIndex, int *gSize, int *distS, int size, bool *change, int *parent)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
 
    while (tid < size)
    {
        int row_begin = gSize[tid];
        int row_end = gSize[tid + 1];
        int min_dist = distS[tid];
        int min_index = parent[tid];

        for(int i = row_begin; i < row_end; i++)
        {            
            if (distS[gIndex[i]] != INT_MAX && distS[gIndex[i]] + g[i] < min_dist)
            {
                min_dist = distS[gIndex[i]] + g[i];
                min_index = gIndex[i];
            }
        }

        //distD[tid] needs to be updated
        distD[tid] = min_dist;
        parent[tid] = min_index;

        if (min_dist < distS[tid])
            change[tid] = true;
        else
            change[tid] = false;

        tid += offset;
    }
}

// modified from cusp
template<int VectorsPerBlock, int ThreadsPerVector>
__global__ void relax_vector(int *distD, int *g, int *gIndex, int *gSize, int *distS, int size, bool *change, int *parent)
{
    __shared__ volatile int cache[VectorsPerBlock * ThreadsPerVector + ThreadsPerVector / 2];
    __shared__ volatile int cacheIndex[VectorsPerBlock * ThreadsPerVector + ThreadsPerVector / 2];
    __shared__ volatile int ptrs[VectorsPerBlock][2];

    int ThreadsPerBlock = VectorsPerBlock * ThreadsPerVector;

    int thread_tid = threadIdx.x + blockIdx.x * ThreadsPerBlock;
    int thread_lane = threadIdx.x & (ThreadsPerVector - 1);
    int vector_id = thread_tid / ThreadsPerVector;
    int vector_lane = threadIdx.x / ThreadsPerVector;
    int num_vectors = VectorsPerBlock * gridDim.x;

    for(int row = vector_id; row < size; row += num_vectors)
    {
        if (thread_lane < 2)
            ptrs[vector_lane][thread_lane] = gSize[row + thread_lane];

        int row_begin = ptrs[vector_lane][0];
        int row_end = ptrs[vector_lane][1];

        int minDist = distS[row];
        int min_index = parent[row];

        if (ThreadsPerVector == 32 && row_end - row_begin > 32)
        {
            int i = row_begin - (row_begin & (ThreadsPerVector - 1)) + thread_lane;
            int d = distS[gIndex[i]];

            if (i >= row_begin && i < row_end && d != INT_MAX && d + g[i] < minDist)
            {
                minDist = d + g[i];
                min_index = gIndex[i];
            }

            for(i += ThreadsPerVector; i < row_end; i += ThreadsPerVector)
            {
                d = distS[gIndex[i]];
                if (d != INT_MAX && d + g[i] < minDist)
                {
                    minDist = d + g[i];
                    min_index = gIndex[i];
                }
            }
        }
        else
        {
            int d;
            for(int i = row_begin + thread_lane; i < row_end; i += ThreadsPerVector)
            {
                d = distS[gIndex[i]];
                if (d != INT_MAX && d + g[i] < minDist)
                {
                    minDist = d + g[i];
                    min_index = gIndex[i];
                }
            }
        }

        cache[threadIdx.x] = minDist;
        cacheIndex[threadIdx.x] = min_index;

        if (ThreadsPerVector > 16)
        {
            if (cache[threadIdx.x + 16] < minDist)
            {
                cache[threadIdx.x] = minDist = cache[threadIdx.x + 16];
                cacheIndex[threadIdx.x] = min_index = cacheIndex[threadIdx.x + 16];
            }
            else
            {
                cache[threadIdx.x] = minDist;
                cacheIndex[threadIdx.x] = min_index;
            }
        }

        if (ThreadsPerVector > 8)
        {
            if (cache[threadIdx.x + 8] < minDist)
            {
                cache[threadIdx.x] = minDist = cache[threadIdx.x + 8];
                cacheIndex[threadIdx.x] = min_index = cacheIndex[threadIdx.x + 8];
            }
            else
            {
                cache[threadIdx.x] = minDist;
                cacheIndex[threadIdx.x] = min_index;
            }
        }

        if (ThreadsPerVector > 4)
        {
            if (cache[threadIdx.x + 4] < minDist)
            {
                cache[threadIdx.x] = minDist = cache[threadIdx.x + 4];
                cacheIndex[threadIdx.x] = min_index = cacheIndex[threadIdx.x + 4];
            }
            else
            {
                cache[threadIdx.x] = minDist;
                cacheIndex[threadIdx.x] = min_index;
            }
        }

        if (ThreadsPerVector > 2)
        {
            if (cache[threadIdx.x + 2] < minDist)
            {
                cache[threadIdx.x] = minDist = cache[threadIdx.x + 2];
                cacheIndex[threadIdx.x] = min_index = cacheIndex[threadIdx.x + 2];
            }
            else
            {
                cache[threadIdx.x] = minDist;
                cacheIndex[threadIdx.x] = min_index;
            }
        }

        if (ThreadsPerVector > 1)
        {
            if (cache[threadIdx.x + 1] < minDist)
            {
                cache[threadIdx.x] = minDist = cache[threadIdx.x + 1];
                cacheIndex[threadIdx.x] = min_index = cacheIndex[threadIdx.x + 1];
            }
            else
            {
                cache[threadIdx.x] = minDist;
                cacheIndex[threadIdx.x] = min_index;
            }
        }

        if (thread_lane == 0)
        {
            distD[row] = cache[threadIdx.x];
            parent[row] = cacheIndex[threadIdx.x];
            if (cache[threadIdx.x] < distS[row])
                change[row] = true;
            else
                change[row] = false;
        }
    }
}
