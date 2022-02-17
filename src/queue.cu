#include <stq/gpu/queue.cuh>
#include <iostream>

// __device__ __host__ Enqueue::Enqueue()
// {
// 	heap_size = 0;
// 	capacity = HEAP_SIZE;
// }

// // Inserts a new key 'k'
// __device__ bool Enqueue::push(const Cell &k)
// { // to avoid overflow, instead of comparing with capacity, we compare with
// capacity -1 	if (heap_size == capacity - 1)
// 	{
// 		return false;
// 	}

// 	// First insert the new key at the end

// 	int i = heap_size;

// 	harr[i] = k;

// 	heap_size++;

// 	return true;
// }

// // Method to remove minimum element (or root) from min heap
// __device__ Cell Enqueue::pop()
// {

// 	Cell root = harr[0];

// 	harr[0] = harr[heap_size - 1];
// 	heap_size--;

// 	MinHeapify();

// 	return root;
// }

// __device__ void Enqueue::MinHeapify()
// {
// 	int itr = 0;

// 	while (itr != heap_size)
//     {
// 			swap(harr[itr], harr[itr+1]);
//             itr++;
//     }
// }

// __device__ bool Enqueue::empty()
// {
// 	return (heap_size == 0);
// }

// __device__ int Enqueue::size()
// {
// 	return heap_size;
// }

// // A utility function to swap two elements
// __device__ void swap(Cell& x, Cell& y)
// {
// 	Cell temp;
// 	temp = x;
// 	x = y;
// 	y = temp;
// }

__device__ __host__ Queue::Queue() {
  heap_size = HEAP_SIZE;
  // capacity = HEAP_SIZE;
}

__device__ int2 Queue::pop() {

  // atomicAdd(&pop_cnt, 1);
  int current = atomicInc(&start, HEAP_SIZE - 1);
  // lock[current].acquire();
  // int2 pair = harr[current];
  // lock[current].release();
  return harr[current];
  ;
  // }
  // } else {
  //   printf("Pop: Failed to lock %i\n", current);
  //   // current, 4000, harr[current].x, itr); current = (current + 1) %
  //   // HEAP_SIZE; itr++;
  //   }
  // unsigned ns = 1000;
  // for (int i=0; i<1000000; i++)
  // 	__nanosleep(ns);
  //   }
  //   int2 ret = make_int2(-1, -1);
  //   return ret; // failure
  // printf("Terminating Pop[%i]\n", curr);
}

__device__ void Queue::push(const int2 pair) {

  int current = atomicInc(&end, HEAP_SIZE - 1);

  // lock[current].acquire();
  harr[current] = pair;
  // atomicAdd(&push_cnt, 1);

  // lock[current].release();
}

__device__ int Queue::size() { return heap_size; }