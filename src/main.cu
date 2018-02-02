///////////////////////////////////////////////////////////////////////////////
// Simple program to add vectors on CPU and GPU
//
// Intended to show common CUDA error checking and usage
///////////////////////////////////////////////////////////////////////////////

#include <cassert>
#include <numeric>
#include <iostream>
#include <vector>

// Check CUDA library calls
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
    do
    {
        if (cudaSuccess != err)
        {
            fprintf(
                stderr,
                "cudaSafeCall() failed at %s:%i : %s\n",
                file,
                line,
                cudaGetErrorString(err)
            );
            exit(-1);
        }
    } while(0);
}

// Check after kernel calls
#define CudaCheckError()  __cudaCheckError(__FILE__, __LINE__)
inline void __cudaCheckError(const char *file, const int line)
{
    do
    {
        cudaError err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(
                stderr,
                "cudaCheckError() failed at %s:%i : %s\n",
                file,
                line,
                cudaGetErrorString(err)
            );
            exit(-1);
        }

        // More careful checking. However, this will affect performance.
        // Comment away if needed.
        err = cudaDeviceSynchronize();
        if( cudaSuccess != err )
        {
            fprintf(
                stderr,
                "cudaCheckError() with sync failed at %s:%i : %s\n",
                file,
                line,
                cudaGetErrorString(err)
            );
            exit(-1);
        }
    } while (0);

    return;
}

// Assert inside kernels
#define CudaAssert(X)                                      \
if (!(X))                                                  \
{                                                          \
    printf( "!!!Thread %d:%d failed assert at %s:%d!!!\n", \
        blockIdx.x, threadIdx.x, __FILE__, __LINE__ );     \
} 

// Utility for looping in kernels
__forceinline__ __device__ int getCurThreadIdx()
{
    const int threadsPerBlock = blockDim.x;
    const int curThreadIdx    = (blockIdx.x * threadsPerBlock) + threadIdx.x;
    return curThreadIdx;
}

// Utility for looping in kernels
__forceinline__ __device__ int getThreadNum()
{
    const int blocksPerGrid   = gridDim.x;
    const int threadsPerBlock = blockDim.x;
    const int threadNum       = blocksPerGrid * threadsPerBlock;
    return threadNum;
}

// Kernel to add two vectors and verify result
__global__ void cudaKernelAddVectors(
    const float *cuInArr1,
    const float *cuInArr2,
    const float *cuRefArr,
    float       *cuOutArr,
    int         N
)
{
    // Iterate input arrays
    for (int i = getCurThreadIdx(); i < N; i += getThreadNum())
    {
        const float v1 = cuInArr1[i];
        const float v2 = cuInArr2[i];
        const float expectedResult = cuRefArr[i];
        const float actualResult = v1 + v2;

        // Let's assert to sanity check
        CudaAssert(expectedResult == actualResult);

        // Sometimes for debugging, it is easier to printf from inside kernel
        if (expectedResult != actualResult)
        {
            printf("Values different! Expected: %f Actual: %f\n", expectedResult, actualResult);
        }

        // Write result to output array
        cuOutArr[i] = actualResult;
    }
}

// Add two vectors in CPU
void cppAddVectors(const std::vector<float> &inVec1, const std::vector<float> &inVec2, std::vector<float> &outVec)
{
    assert(inVec1.size() == inVec2.size());
    assert(inVec1.size() == outVec.size());

    for (int i = 0; i < inVec1.size(); ++i)
    {
        outVec.at(i) = inVec1.at(i) + inVec2.at(i);
    }
}

int main()
{
    ////
    // Compute on CPU in C++
    ////

    const int N = 100;
    std::vector<float> inVec1(N);
    std::vector<float> inVec2(N);
    std::vector<float> refVec(N);

    std::iota(inVec1.begin(), inVec1.end(), 0);
    std::iota(inVec2.begin(), inVec2.end(), 99);

    cppAddVectors(inVec1, inVec2, refVec);

    ////
    // Move data to GPU
    ////
    
    float *cuInArr1;
    float *cuInArr2;
    float *cuRefArr;
    float *cuOutArr;

    const int bytesN = N * sizeof(float);
    CudaSafeCall(cudaMalloc(&cuInArr1, bytesN));
    CudaSafeCall(cudaMalloc(&cuInArr2, bytesN));
    CudaSafeCall(cudaMalloc(&cuRefArr, bytesN));
    CudaSafeCall(cudaMalloc(&cuOutArr, bytesN));

    CudaSafeCall(cudaMemcpy(cuInArr1, inVec1.data(), bytesN, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(cuInArr2, inVec2.data(), bytesN, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(cuRefArr, refVec.data(), bytesN, cudaMemcpyHostToDevice));

    ////
    // Compute and cross-check on GPU
    ////

    const int blocksPerGrid = 64;
    const int threadsPerBlock = 256;
    cudaKernelAddVectors<<<blocksPerGrid, threadsPerBlock>>>(
        cuInArr1,
        cuInArr2,
        cuRefArr,
        cuOutArr,
        N
    );
    CudaCheckError();

    return 0;
}
