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
	// int itr = 0;
	int current = curr % HEAP_SIZE;
	// if (current == 0) current = HEAP_SIZE - 1;
	// else current = current - 1;
	// int old  = current;
	while (1)
	{
		// old  = current;
		// if (lock[current].try_acquire() )
		if (harr[current].x >= 0.0 )
		{
			lock[current].acquire();
			if (harr[current].x < 0)
			{
				// printf("Pop[%i]: adding 1 to %i/%i\n", curr, current, HEAP_SIZE);
				// current = (current + 1) % HEAP_SIZE;
				// itr++;
				lock[current].release();
			}
			else 
			{
				int2 pair = harr[current];
				harr[current].x = -1; //set as removed
				// printf("Pop[%i]: pair (%i,%i) from harr[%i] with cap %i\n", curr, pair.x, pair.y, current, HEAP_SIZE);
				lock[current].release();
				return pair;
			}
		}
		else 
		{
			// printf("Pop[%i]: Failed to lock %i (%i) w/ val %i, itr %i\n", curr, current, 4000, harr[current].x, itr);
			// current = (current + 1) % HEAP_SIZE;
			// itr++;
		}
		// unsigned ns = 1000;
		// for (int i=0; i<1000000; i++)
		// 	__nanosleep(ns);

	}
	int2 ret = make_int2(-1,-1);
	return ret; //failure
	// printf("Terminating Pop[%i]\n", curr);
	
}

__device__ int Queue::push(int tid, int2 pair)
{
	int current = tid % 4000;
	// int old  = current;
	while (1)
	{
		// old = current;
		// if (lock[current].try_acquire() )
		if (harr[current].x < 0)
		{
			lock[current].acquire();
			if (harr[current].x >= 0)
			{
				// printf("Push[%i]: adding 1 to %i/%i\n", tid, current, 4000);
				// current = (current + 1) % 4000;
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
			// printf("Push[%i]: Failed to lock %i (%i) w/ val %i\n", tid, current, 4000, harr[current].x);
			// current = (current + 1) % 4000;
		}
		// unsigned ns = 1000;
		// for (int i=0; i<1000000; i++)
		// 	__nanosleep(ns);

	}
	// printf("Terminating Push[%i]\n", tid);
}

__device__ int Queue::size()
{
	return heap_size;
}