#pragma once

#include <array>
#include <gpf/ids.hpp>
#include <vector>
#include <gpf/mesh.hpp>

namespace tet_mesh {

struct Tet {
    std::array<gpf::VertexId, 4> vertices;
    std::array<std::size_t, 4> faces;
    std::array<gpf::EdgeId, 6> edges;
};

struct VertexProp {
    std::array<double, 3> pt;
    std::vector<double> distances;
};

struct FaceProp {
    std::array<std::size_t, 2> cells;
};

using TetMesh = gpf::SurfaceMesh<VertexProp, gpf::Empty, gpf::Empty, FaceProp>;
}
