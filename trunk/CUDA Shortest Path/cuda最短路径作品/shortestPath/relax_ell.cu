__global__ void relax_ell(int *d_dst, int *Cv, int *Cj, int *d_src, int num_rows, int num_cols_per_row)
{
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	while (row < num_vertices)
	{
		int min_dist = INT_MAX;

		for(int i = 0; i < num_cols_per_row; i++)
		{
			int col = Cj[num_rows * i + row];
			int val = Cv[num_rows * i + row];

			if (col != -1)
				min_dist = min(min_dist, val + d_src[col]);
		}

		d_dst[row] = min_dist;

		row += offset;
	}
}

