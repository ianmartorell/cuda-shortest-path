#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ell_bellman.cuh"
using namespace std;
using namespace thrust;

void test_ell_bellman(int source, int target, string filename, bool printPath)
{
    cout << "Reading Graph..." << endl;

    ReadELLGraph(filename);

    if (target == -1)
        target = g_size - 1;

    cout << "Doing ShortestPath Algorithm..." << endl;

    CUDATimer timer;

    int dist = ell_bellman(source, target, g_size);

    string method = "CUDA ELL Bellman";

    PrintResult(method, source, target, dist, timer.seconds_elapsed(), printPath);

    free(g);
    free(gIndex);
    free(parent);
    cudaFree(dev_g);
    cudaFree(dev_gIndex);
}

int ell_bellman(int source, int target, int size)
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
    const int blocksPerGrid = min(prop.maxGridSize[0], (size + threadsPerBlock - 1) / threadsPerBlock);

    while (true)
    {
        if (flag)
        {
            relax<<<blocksPerGrid, threadsPerBlock>>>(dev_dist2_ptr, dev_g, dev_gIndex, dev_dist1_ptr, size, num_cols_per_row, dev_change_ptr, dev_parent_ptr);
        }
        else
        {
            relax<<<blocksPerGrid, threadsPerBlock>>>(dev_dist1_ptr, dev_g, dev_gIndex, dev_dist2_ptr, size, num_cols_per_row, dev_change_ptr, dev_parent_ptr);
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

__global__ void relax(int *distD, int *g, int *gIndex, int *distS, int size, int num_cols_per_row, bool *change, int *parent)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (row < size)
    {
        int min_dist = distS[row];
        int min_index = parent[row];
        
        for(int i = 0, j = 0; i < num_cols_per_row; i++, j += size)
        {
            int col = gIndex[j + row];
            int val = g[j + row];

            if (col != -1 && distS[col] != INT_MAX && val + distS[col] < min_dist)
            {
                min_dist = val + distS[col];
                min_index = col;
            }
        }

        distD[row] = min_dist;
        parent[row] = min_index;

        if (min_dist < distS[row])
            change[row] = true;
        else
            change[row] = false;

        row += offset;
    }
}
