__global__ void relax_csr_vector(int *d_dst, int *Cv, int *Cj, int *Cp, int *d_src, int num_rows)
{
	__shared__ volatile int min_dist[];

	int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
	int thread_lane = threadIdx.x & (32 - 1);
	int warp_id = thread_id / 32;
	int warp_lane = threadIdx.x / 32;
	int num_warps = (blockDim.x / 32) * gridDim.x;

	int row = warp_id;

	while (row < num_rows)
	{
		int row_begin = Cp[row];
		int row_end = Cp[row+1];

		int min_dist[threadIdx.x] = INT_MAX;

		for(int i = row_begin + lane; i < row_end; i++)
			min_dist = min(min_dist, d_src[Cj[i]] + Cv[i]);

		if (lane < 16) min_dist[threadIdx.x] = min(min_dist[threadIdx.x], min_dist[threadIdx.x + 16]);
		if (lane < 8) min_dist[threadIdx.x] = min(min_dist[threadIdx.x], min_dist[threadIdx.x + 8]);
		if (lane < 4) min_dist[threadIdx.x] = min(min_dist[threadIdx.x], min_dist[threadIdx.x + 4]);
		if (lane < 2) min_dist[threadIdx.x] = min(min_dist[threadIdx.x], min_dist[threadIdx.x + 2]);
		if (lane < 1) min_dist[threadIdx.x] = min(min_dist[threadIdx.x], min_dist[threadIdx.x + 1]);

		if (thread_lane == 0)
			d_dst[row] = min(d_dst[row], min_dist[threadIdx.x]);

		row += num_warps;
	}
}

