#pragma once

// #include <gpubf/object.cuh>

// https://github.com/wangbolun300/GPUTI/blob/master/src/queue.cu

static const int HEAP_SIZE=5;

using namespace std;

// Prototype of a utility function to swap two integers
// __device__ void swap(Cell&x, Cell &y);


// class Enqueue
// {
// 	Cell harr[HEAP_SIZE]; // pointer to array of elements in heap
// 	int capacity; // maximum possible size of min heap
// 	int heap_size; // Current number of elements in min heap
//     Cell root;// temporary variable used for extractMin()

// public:
// 	// Constructor
//    __device__ __host__ Enqueue();

// 	// to heapify a subtree with the root at given index
// 	__device__ void MinHeapify();
// 	__device__ bool empty();

// 	// to extract the root which is the minimum element
// 	__device__ Cell pop();

// 	// Inserts a new key 'k'
// 	__device__ bool push(const Cell &k);

//     __device__ int size();
// };

__device__ __host__ class Queue
{
public:
	int2 harr[HEAP_SIZE]; // pointer to array of elements in heap
	int capacity; // maximum possible size of min heap
	int heap_size; // Current number of elements in min heap
    // Cell root;// temporary variable used for extractMin()


	// Constructor
   __device__ __host__ Queue();

	__device__ int2 pop();

	// Inserts a new key 'k'
	__device__ void push(int2 pair);

    __device__ int size();
};