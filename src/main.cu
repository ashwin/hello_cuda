#include <thrust/device_vector.h>
#include <thrust/sequence.h>

__global__ void fooKernel(const int* i_arr, int* o_arr, int N)
{
    const int threadNum     = gridDim.x * blockDim.x;
    const int curThreadIdx  = (blockIdx.x * blockDim.x) + threadIdx.x;
 
    for (int i = curThreadIdx; i < (N - 1); i += threadNum)
        o_arr[i] = i_arr[i] + i_arr[i + 1];
}
 
int main()
{
    const int N = 10000;
    thrust::device_vector<int> i_vec(N), o_vec(N);

    thrust::sequence(i_vec.begin(), i_vec.end());

    fooKernel<<<128, 128>>>(
        thrust::raw_pointer_cast(&i_vec[0]),
        thrust::raw_pointer_cast(&o_vec[0]),
        N);

    for (int i = N - 10; i < N; ++i)
        std::cout << o_vec[i] << std::endl;
 
    return 0;
}
