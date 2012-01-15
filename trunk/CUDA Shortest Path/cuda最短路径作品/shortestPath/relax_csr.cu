__global__ void relax_csr_scalar(int *d_dst, int *Cv, int *Cj, int *Cp, int *d_src, int num_vertices)
{
    int vertice_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    while (vertice_idx < num_vertices)
    {
        int vertice_begin = Cp[vertice_idx];
        int vertice_end = Cp[vertice_idx+1];

        int min_dist = INT_MAX;

        for(int i = vertice_begin; i < vertice_end; i++)
            min_dist = min(min_dist, d_src[Cj[i]] + Cv[i]);

        d_dst[vertice_idx] = min_dist;

		vertice_id += offset;
    }
}

