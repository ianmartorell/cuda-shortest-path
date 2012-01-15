#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "dijkstra.cuh"
using namespace std;
using namespace thrust;

struct smaller_dist
{
    __device__ tuple<int, int> operator()(const tuple<int, int> &a, const tuple<int, int> &b)
    {
        if (get<0>(a) < get<0>(b))
            return a;
        else
            return b;
    }
};

void test_dijkstra(int source, int target, string filename, bool printPath)
{
    cout << "Reading Graph..." << endl;

    ReadGraph(filename);

    if (target == -1)
        target = g_size - 1;

    cout << "Doing ShortestPath Algorithm..." << endl;

    CUDATimer timer;

    int dist = dijkstra(source, target);

    string method = "CUDA Dijkstra";

    PrintResult(method, source, target, dist, timer.seconds_elapsed(), printPath);

    free(g);
    free(gSize);
    free(gIndex);
    free(parent);
    cudaFree(dev_g);
    cudaFree(dev_gSize);
    cudaFree(dev_gIndex);
}

int dijkstra(int source, int target)
{    
    host_vector<int> h_dist(g_size, INT_MAX);
    h_dist[source] = 0;

    device_vector<int> d_dist(h_dist);

    int *d_dist_ptr = raw_pointer_cast(&d_dist[0]);

    device_vector<int> dev_parent(g_size, -1);
    int *dev_parent_ptr = raw_pointer_cast(&dev_parent[0]);

    bool *dev_canUse;
    cudaMalloc(&dev_canUse, sizeof(bool) * g_size);
    cudaMemset(dev_canUse, true, sizeof(bool) * g_size);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int threadsPerBlock = prop.maxThreadsDim[0];
    const int blocksPerGrid = std::min(prop.maxGridSize[0], (g_size + threadsPerBlock - 1) / threadsPerBlock);

    int minIndex, minDist;
    tuple<int, int> node;

    for(int i = 0; i < g_size - 1; i++)
    {
        node = ExtractMin(d_dist);

        minIndex = get<1>(node); // extract the min node index
        minDist = get<0>(node); // dist

        // no node needs to be update
        if (minDist == INT_MAX)
            break;

        //update host dist
        h_dist[minIndex] = minDist;

        relax<<<blocksPerGrid, threadsPerBlock>>>(d_dist_ptr, dev_canUse, minDist, dev_g, dev_gIndex, minIndex, gSize[minIndex], gSize[minIndex + 1] - gSize[minIndex], dev_parent_ptr);
    }

    cudaMemcpy(parent, dev_parent_ptr, sizeof(int) * g_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_canUse);

    int ret = h_dist[target];

    return ret;
}

tuple<int, int> ExtractMin(device_vector<int> &dist)
{
    counting_iterator<int> begin(0);
    counting_iterator<int> end(dist.size());

    tuple<int, int> init(INT_MAX, 0);

    tuple<int, int> smallest = reduce(make_zip_iterator(make_tuple(dist.begin(), begin)),
        make_zip_iterator(make_tuple(dist.end(), end)), init, smaller_dist());

    return smallest;
}

__global__ void relax(int *dist, bool *canUse, int minDist, int *g, int *gIndex, int index, int beg, int size, int *parent)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    if (tid == 0)
    {
        canUse[index] = false; // minDist[index] is finded
        dist[index] = INT_MAX; // make sure node index will not be extracted again in ExtractMin
    }

    while (tid < size)
    {
        int idx = beg + tid; // find index in gIndex
        int node = gIndex[idx];
        if (canUse[node])
        {
            if (minDist + g[idx] < dist[node])
            {
                dist[node] = minDist + g[idx];
                parent[node] = index;
            }
        }
        tid += offset;
    }
}