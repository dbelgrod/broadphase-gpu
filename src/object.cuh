#pragma once


__device__ struct Cell {
    float3 min;
    float3 max;
    Aabb past;
    Aabb * boxes; 
    int Nboxes;
    int id;

    __device__ Cell() = default;

    __device__ Cell(Aabb * boxes__x, Aabb * boxes__y, int N)
    {
        min = {boxes__x[0].min.x , boxes__y[0].min.y, -1};
        max = {boxes__x[N-1].max.x , boxes__y[N-1].max.y, -1};
        // past = NULL;
        boxes = boxes__x;
        Nboxes = N;
        id = 0;
    };

    __device__ Cell(float left, float right, float upper, float lower, Aabb * d_boxes, int N , int gid) //left, right, upper, lower
    {
        //cell should know its vertices, boxes, 
        min = {left, lower, -1};
        max = {right, upper, -1};
        boxes = d_boxes;
        Nboxes = N;
        id = gid;
        // past = parent->past;
        // boxes = parent->boxes;
        // Nboxes = 0;
        
        
        
        // for (size_t i = 0; i < parent->Nboxes; i++)
        // {
        //     if (parent->boxes[i].min.x > left && parent->boxes[i].max.x < right && 
        //          parent->boxes[i].min.y > lower && parent->boxes[i].max.y < upper)
        //     {
        //         boxes[Nboxes] = parent->boxes[i];
        //         Nboxes++;
        //     }
        // }
    };

    __device__ float Norm(){
        return sqrt((max.x-min.x)*(max.x-min.x) + (max.y-min.y)*(max.y-min.y));
    };

    __device__ void Simplify(int * count, int2 * overlaps, int G)
    {
        for (int i = 0; i < Nboxes; i++)
        {
            // pr intf("%i\n", i);
            // "bad" box with 1+ vertices not in cell
            if (boxes[i].min.x < min.x && boxes[i].max.x > max.x)
                {
                    // scan down all boxes
                    float shrink = boxes[i].max.y - boxes[i].min.y;
                    for (int j = 0; j < Nboxes; j++)
                    {
                        // if (boxes[j].max.x >= boxes[i].min.x && boxes[j].max.x >= boxes[i].min.x )
                        if (does_collide(&boxes[i], &boxes[j]))
                            add_overlap(i, j, count, overlaps, G);
                            // add_overlap(i, j);

                        // shrink boxes lower than "bad" box + bad box itself
                        if (boxes[j].max.y < boxes[i].min.y)
                            boxes[j].min.y += shrink;
                        
                        
                    }
                    // boxes[i].min.y = boxes[i].max.y;
                }
        }
    };

    // 
    __device__ void Cut(Cell* nextcells)
    {
        //get the middle x, cut it halfway on y and at its points
        past = boxes[Nboxes/2];
        // Cell cellLU(this, min.x, past.min.x, max.y, (past.max.y-past.min.y)/2); //Cell, Left, Right, Upper, Lower
        // Cell cellLL(this, min.x, past.min.x, (past.max.y-past.min.y)/2, min.y);
        // Cell cellRU(this, past.max.x, max.x, max.y, (past.max.y-past.min.y)/2);
        // Cell cellRL(this, past.max.x, max.x, (past.max.y-past.min.y)/2, min.y);

        // Cell(float left, float right, float upper, float lower, Aabb * d_boxes, int N )

        Aabb * lboxes = &boxes[0];
        Aabb * rboxes = &boxes[Nboxes/2];

        int N = Nboxes/2 + 1;

        Cell cellL(min.x, (past.max.x-past.min.x)/2, max.y, min.y, lboxes, N, 2*id+1);
        Cell cellR((past.max.x-past.min.x)/2, max.x, max.y, min.y, rboxes, N, 2*id+2);

        // Cell cellOL(this, min.x, past.min.x, max.y, min.y);
        // Cell cellOR(this, past.max.x, max.x max.y, min.y);

        // Cell cellIU(this, past.min.x, past.max.x, max.y, (past.max.y-past.min.y)/2);
        // Cell cellIL(this, past.min.x, past.max.x, (past.max.y-past.min.y)/2, min.y);
        // enqueue.push(cellLU);
        // enqueue.push(cellLL);
        // enqueue.push(cellRU);
        // enqueue.push(cellRL);
        nextcells[0] = cellL;
        nextcells[1] = cellR;
        return;
    };

};