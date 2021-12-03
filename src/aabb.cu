#include <gpubf/aabb.cuh>

__host__ __device__ bool is_face(const Aabb& x)
{
    return x.vertexIds.z >= 0;
};

__host__ __device__ bool is_face(const int3& vids)
{
    return vids.z >= 0;
};

__host__ __device__ bool is_edge(const Aabb& x)
{
    return x.vertexIds.z < 0 && x.vertexIds.y >= 0 ;
};

__host__ __device__ bool is_edge(const int3& vids)
{
    return vids.z < 0 && vids.y >= 0 ;
};

__host__ __device__ bool is_vertex(const Aabb& x)
{
    return x.vertexIds.z < 0  && x.vertexIds.y < 0;
};

__host__ __device__ bool is_vertex(const int3& vids)
{
    return vids.z < 0  && vids.y < 0;
};

__host__ __device__ bool is_valid_pair(const Aabb& a, const Aabb& b)
{
    return (is_vertex(a) && is_face(b)) ||
        (is_face(a) && is_vertex(b)) ||
        (is_edge(a) && is_edge(b));
};

__host__ __device__ bool is_valid_pair(const int3& a, const int3& b)
{
    return (is_vertex(a) && is_face(b)) ||
        (is_face(a) && is_vertex(b)) ||
        (is_edge(a) && is_edge(b));
};

double nextafter_up(double x){return nextafter(x, x+1.);};
double nextafter_down(double x){return nextafter(x, x-1.);};

void addEdges
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    Eigen::MatrixXi& edges, 
    vector<Aabb>& boxes
)
{
    for (unsigned long i = 0; i < edges.rows(); i++) {
        Eigen::MatrixXd edge_vertex0_t0 = vertices_t0.row(edges(i, 0));
        Eigen::MatrixXd edge_vertex1_t0 = vertices_t0.row(edges(i, 1));
        Eigen::MatrixXd edge_vertex0_t1 = vertices_t1.row(edges(i, 0));
        Eigen::MatrixXd edge_vertex1_t1 = vertices_t1.row(edges(i, 1));

        Eigen::MatrixXd points(4, edge_vertex0_t0.size());
        points.row(0) = edge_vertex0_t0;
        points.row(1) = edge_vertex1_t0;
        points.row(2) = edge_vertex0_t1;
        points.row(3) = edge_vertex1_t1;

        int vertexIds[3] = {edges(i,0), edges(i,1), -edges(i,0)-1};

        Eigen::MatrixXf lower_bound = points.colwise().minCoeff().unaryExpr(&nextafter_down).cast<float>();
        Eigen::MatrixXf upper_bound = points.colwise().maxCoeff().unaryExpr(&nextafter_up).cast<float>();
        boxes.emplace_back(boxes.size(), i, Simplex::EDGE, vertexIds, lower_bound.array().data(), upper_bound.array().data());
    }
}

void addVertices
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    vector<Aabb>& boxes
)
{
    for (unsigned long i = 0; i < vertices_t0.rows(); i++) {
        Eigen::MatrixXd vertex_t0 = vertices_t0.row(i);
        Eigen::MatrixXd vertex_t1 = vertices_t1.row(i);

        Eigen::MatrixXd points(2, vertex_t0.size());
        points.row(0) = vertex_t0;
        points.row(1) = vertex_t1;

        int vertexIds[3] = {i, -i-1, -i-1};

        Eigen::MatrixXf lower_bound = points.colwise().minCoeff().unaryExpr(&nextafter_down).cast<float>();
        Eigen::MatrixXf upper_bound = points.colwise().maxCoeff().unaryExpr(&nextafter_up).cast<float>();
        boxes.emplace_back(boxes.size(), i, Simplex::VERTEX, vertexIds, lower_bound.array().data(), upper_bound.array().data());
    }
}

void addFaces
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    Eigen::MatrixXi& faces, 
    vector<Aabb>& boxes
)
{
    for (unsigned long i = 0; i < faces.rows(); i++) {
        Eigen::MatrixXd face_vertex0_t0 = vertices_t0.row(faces(i, 0));
        Eigen::MatrixXd face_vertex1_t0 = vertices_t0.row(faces(i, 1));
        Eigen::MatrixXd face_vertex2_t0 = vertices_t0.row(faces(i, 2)); 
        Eigen::MatrixXd face_vertex0_t1 = vertices_t1.row(faces(i, 0));
        Eigen::MatrixXd face_vertex1_t1 = vertices_t1.row(faces(i, 1)); 
        Eigen::MatrixXd face_vertex2_t1 = vertices_t1.row(faces(i, 2));

        Eigen::MatrixXd points(6, face_vertex0_t0.size());
        points.row(0) = face_vertex0_t0;
        points.row(1) = face_vertex1_t0;
        points.row(2) = face_vertex2_t0;
        points.row(3) = face_vertex0_t1;
        points.row(4) = face_vertex1_t1;
        points.row(5) = face_vertex2_t1;

        int vertexIds[3] = {faces(i,0), faces(i,1), faces(i,2)};

        Eigen::MatrixXf lower_bound = points.colwise().minCoeff().unaryExpr(&nextafter_down).cast<float>();
        Eigen::MatrixXf upper_bound = points.colwise().maxCoeff().unaryExpr(&nextafter_up).cast<float>();;
        boxes.emplace_back(boxes.size(), i, Simplex::FACE, vertexIds, lower_bound.array().data(), upper_bound.array().data());
    }
};

