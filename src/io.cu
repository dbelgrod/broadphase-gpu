#include <gpubf/io.cuh>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/edges.h>

void constructBoxes
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    Eigen::MatrixXi& faces, 
    Eigen::MatrixXi& edges, 
    vector<Aabb>& boxes
)
{
    addVertices(vertices_t0, vertices_t1, boxes);
    addEdges(vertices_t0, vertices_t1, edges, boxes);
    addFaces(vertices_t0, vertices_t1, faces, boxes);
}

void parseMesh(
    const char* filet0, 
    const char* filet1, 
    Eigen::MatrixXd& V0, 
    Eigen::MatrixXd& V1, 
    Eigen::MatrixXi& F, 
    Eigen::MatrixXi& E)
{
    // Eigen::MatrixXd V0;
    // Eigen::MatrixXd V1;
    // Eigen::MatrixXi F;

    string fn = string(filet0);
    string ext = fn.substr(fn.rfind('.') + 1);

    if(ext == "obj") 
    {
        igl::readOBJ(filet0, V0, F);
        igl::readOBJ(filet1, V1, F);
    }
    else 
    {
        igl::readPLY(filet0, V0, F);
        igl::readPLY(filet1, V1, F);
    }

    
    // Eigen::MatrixXi E;
    igl::edges(F,E);
    // faces should be same F^{t=0} = F^{t=1}
    // constructBoxes(V0, V1, F, E, boxes);
}