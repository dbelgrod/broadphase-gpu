// goal is to understand global/shared memory access and bank conflicts
#include <stdio.h>
#include <getopt.h>

int vflag = 0;
int vprintf(const char *format, ...) {
  if (vflag)
    printf(format);
  return 1;
}

__global__ struct Fbox {
  int id;
  int finishTime;
  int sharedTime;
  int bank;
  float2 buffer;
};

__global__ void reverse(Fbox *d_nums, Fbox *d_rev, int N, int B,
                        int *sharedTime, int *finishTime) {
  extern __shared__ Fbox s_nums[];
  int g_tid = threadIdx.x + blockIdx.x * blockDim.x * B;
  int l_tid = threadIdx.x;

  if (g_tid >= N)
    return;

  for (int i = 0; i < B; i++) {
    s_nums[l_tid + blockDim.x * i] = d_nums[g_tid + blockDim.x * i];
    int j = atomicAdd(sharedTime, 1);
    s_nums[l_tid + blockDim.x * i].sharedTime = j;
    s_nums[l_tid + blockDim.x * i].bank = (l_tid + blockDim.x * i) % 32;
  }

  for (int i = 0; i < B; i++) {
    // d_rev[N - (g_tid + blockDim.x*i)] = s_nums[l_tid + blockDim.x*i];
    d_rev[g_tid + blockDim.x * i] = s_nums[l_tid + blockDim.x * i];
    int k = atomicAdd(finishTime, 1);
    d_rev[g_tid + blockDim.x * i].finishTime = k;
  }
}

__global__ void resetTime(int *time) { *time = 0; }

int main(int argc, char **argv) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int c;
  while ((c = getopt(argc, argv, "v:")) != -1) {
    switch (c) {
    case 'v':
      vflag = 1;
      break;
    }
  }

  int N;
  int B;
  if (argc == 2) {
    N = atoi(argv[argc - 1]);
    B = 1;
  } else if (argc >= 3) {
    N = atoi(argv[argc - 2]);
    B = atoi(argv[argc - 1]);
  } else
    abort();

  Fbox nums[N];
  Fbox rev[N];

  for (int i = 0; i < N; i++) {
    nums[i].id = i;
    rev[i].id = 0;
  }
  printf("Fbox size: %i B\n", sizeof(Fbox));
  Fbox *d_nums;
  Fbox *d_rev;
  cudaMalloc((void **)&d_nums, sizeof(Fbox) * N);
  cudaMemcpy(d_nums, nums, sizeof(Fbox) * N, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_rev, sizeof(Fbox) * N);

  int *sharedTime;
  int *finishTime;
  cudaMalloc((void **)&sharedTime, sizeof(int));
  cudaMalloc((void **)&finishTime, sizeof(int));
  resetTime<<<1, 1>>>(sharedTime);
  resetTime<<<1, 1>>>(finishTime);

  int BLOCK_SIZE = 1024;
  int grid_size = N / BLOCK_SIZE / B + 1;

  int ITER = 1;
  float milliseconds = 0;
  float avg_ms = 0;
  for (int i = 0; i < ITER; i++) {
    cudaEventRecord(start);
    reverse<<<grid_size, BLOCK_SIZE, 49152>>>(d_nums, d_rev, N, B, sharedTime,
                                              finishTime);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    avg_ms += milliseconds;
  }
  avg_ms /= ITER;
  printf("Avg. elapsed time: %.6f ms\n", avg_ms);

  cudaMemcpy(rev, d_rev, sizeof(Fbox) * N, cudaMemcpyDeviceToHost);
  long long sum = 0;
  vprintf("shared -> ");
  for (int i = 0; i < N; i++) {
    sum += rev[i].id;
    vprintf("(%i:%i) ", rev[i].id, rev[i].sharedTime);
  }
  vprintf("\nfinish -> ");
  for (int i = 0; i < N; i++) {
    vprintf("(%i:%i) ", rev[i].id, rev[i].finishTime);
  }

  vprintf("\nbank -> ");
  for (int i = 0; i < N; i++) {
    vprintf("(%i:%i) ", rev[i].id, rev[i].bank);
  }

  vprintf("\n");
  printf("sum -> %llu\n", sum);
  printf("Grid size: %i\n", grid_size);
}
