// #include <gpubf/klee.cuh>
#include <gpubf/queue.cuh>
#include <iostream>



// __device__ __host__ Enqueue::Enqueue()
// {
// 	heap_size = 0;
// 	capacity = HEAP_SIZE;
// }


// // Inserts a new key 'k'
// __device__ bool Enqueue::push(const Cell &k)
// { // to avoid overflow, instead of comparing with capacity, we compare with capacity -1
// 	if (heap_size == capacity - 1)
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

__device__ __host__ Queue::Queue()
{
	heap_size = 0;
	capacity = HEAP_SIZE;
}

__device__ int2 Queue::pop(int curr)
{
	// skip while data inside or locked
	int current = curr;
	while (1)
	{
		if (lock[current].try_acquire())
		{
			if (harr[current].x < 0)
			{
				// printf("Pop: adding 1 to %i/%i\n", current, heap_size);
				current = (current + 1) % heap_size;
				lock[current].release();
			}
			else 
			{
				int2 pair = harr[current];
				harr[current].x = -1; //set as removed
				// printf("Pop: pair (%i,%i) from harr[%i] with cap %i\n", pair.x, pair.y, current, capacity);
				lock[current].release();
				return pair;
			}
		}
		else 
		{
			// printf("Pop: Failed to lock, adding 1 to %i/%i\n", current, heap_size);
			current = (current + 1) % heap_size;
		}
		unsigned ns = 1000;
		// for (int i=0; i<1000000; i++)
			// __nanosleep(ns);
	}
	
}

__device__ int Queue::push(int tid, int2 pair)
{
	int current = tid % HEAP_SIZE;
	while (1)
	{
		
		if (lock[current].try_acquire() )
		{
			if (harr[current].x >= 0)
			{
				// printf("Push[%i]: adding 1 to %i/%i\n", tid, current, heap_size);
				current = (current + 1) % heap_size;
				lock[current].release();
			}
			else 
			{
				harr[current] = pair;
				// printf("Push[%i]: pair (%i,%i) to harr[%i]\n", tid, pair.x, pair.y, current);
				lock[current].release();
				return current;
			}
		}
		else 
		{
			// printf("Push[%i]: Failed to lock for, adding 1 to %i/%i\n", tid, current, heap_size);
			current = (current + 1) % heap_size;
		}
		unsigned ns = 1000;
		// for (int i=0; i<1000000; i++)
			// __nanosleep(ns);

	}
}

__device__ int Queue::size()
{
	return heap_size;
}