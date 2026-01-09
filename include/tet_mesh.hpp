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

    [[nodiscard]] static auto edge_index(const std::size_t pa, const std::size_t pb) noexcept -> std::size_t {
        const auto min_idx = pa < pb ? pa : pb;
        return (min_idx != 0 ? 0 : 1) + 5 - pa - pb;
    }
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
