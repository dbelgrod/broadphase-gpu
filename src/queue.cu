#include <gpubf/klee.cuh>
#include <gpubf/queue.cuh>



__device__ __host__ Enqueue::Enqueue()
{
	heap_size = 0;
	capacity = HEAP_SIZE;
}

// Inserts a new key 'k'
__device__ bool Enqueue::push(const Cell &k)
{ // to avoid overflow, instead of comparing with capacity, we compare with capacity -1
	if (heap_size == capacity - 1)
	{
		return false;
	}

	// First insert the new key at the end

	int i = heap_size;

	harr[i] = k;

	heap_size++;

	return true;
}

// Method to remove minimum element (or root) from min heap
__device__ Cell Enqueue::pop()
{

	Cell root = harr[0];

	harr[0] = harr[heap_size - 1];
	heap_size--;

	MinHeapify();

	return root;
}

__device__ void Enqueue::MinHeapify()
{
	int itr = 0;

	while (itr != heap_size)
    {
			swap(harr[itr], harr[itr+1]);
            itr++;
    }
}

__device__ bool Enqueue::empty()
{
	return (heap_size == 0);
}

__device__ int Enqueue::size()
{
	return heap_size;
}

// A utility function to swap two elements
__device__ void swap(Cell& x, Cell& y)
{
	Cell temp;
	temp = x;
	x = y;
	y = temp;
}
