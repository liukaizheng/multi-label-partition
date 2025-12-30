#include "material_interface.hpp"
#include "gpf/mesh.hpp"
#include "gpf/surface_mesh.hpp"
#include "tet_mesh.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <ranges>
#include <unordered_set>

namespace ranges = std::ranges;
namespace views = std::views;

struct VertexProp {
    std::array<std::size_t, 4> materials;
};

struct EdgeProp {
    std::array<std::size_t, 3> materials;
};

struct FaceProp {
    std::array<std::size_t, 2> materials{{gpf::kInvalidIndex, gpf::kInvalidIndex}};
};

struct Cell {
    std::vector<std::size_t> faces;
};

using Mesh = gpf::SurfaceMesh<VertexProp, gpf::Empty, EdgeProp, FaceProp>;

struct MaterialInterface {
    Mesh mesh;
    std::vector<Cell> cells;
    std::vector<std::array<double, 4>> materials;

    MaterialInterface();
    MaterialInterface(const MaterialInterface&) = default;

    void compute_vert_orientations(const std::size_t mid);
    void add_material(const std::array<double, 4>& material);
};

MaterialInterface::MaterialInterface() {
    mesh = Mesh::new_in(std::vector<std::vector<std::size_t>> {{2, 1, 3}, {0 ,2, 3}, {1, 0, 3}, {0, 1, 2}});
    for (auto v : mesh.vertices()) {
        auto& material = v.data().property.materials;
        for (auto [m, fid] : ranges::zip_view{material, v.incoming_halfedges() | views::transform([](auto h) { return h.face().id.idx; })}) {
            m = fid;
        }
        material.back() = 4;
    }

    for (auto e : mesh.edges()) {
        auto& material = e.data().property.materials;
        for (auto [m, fid] : ranges::zip_view{material, e.halfedges() | views::transform([](auto h) { return h.face().id.idx; })}) {
            m = fid;
        }
        material.back() = 4;
    }

    cells = { Cell{.faces{0, 2, 4, 6}} };
}

void MaterialInterface::add_material(const std::array<double, 4>& material) {
    const auto mid = materials.size();
    materials.emplace_back(material);
    if (materials.size() == 1) {
        return;
    }
}

void MaterialInterface::compute_vert_orientations(const std::size_t mid) {

}

void do_material_interface(
    const std::vector<tet_mesh::Tet> &tets,
    const tet_mesh::TetMesh &tet_mesh
) {
    MaterialInterface base_mi;
    const auto n_materials = tet_mesh.vertex(gpf::VertexId{0}).data().property.distances.size();

    std::vector<std::size_t> separators{0};
    std::vector<std::size_t> v_high_materials;
    for (const auto v : tet_mesh.vertices()) {
        const auto& v_materials = v.data().property.distances;
        const auto max_idx = std::distance(v_materials.begin(), std::ranges::max_element(v_materials));
        v_high_materials.emplace_back(max_idx);
        for (std::size_t i = 0; i < n_materials; i++) {
            if (i != max_idx && v_materials[i] == v_materials[max_idx]) {
                v_high_materials.emplace_back(i);
            }
        }
        separators.emplace_back(v_high_materials.size());
    }

    for (const auto &tet : tets) {
        std::unordered_set<std::size_t> tet_materials;
        for (const auto vid : tet.vertices) {
            const auto idx = vid.idx;
            tet_materials.insert(v_high_materials.begin() + separators[idx], v_high_materials.begin() + separators[idx + 1]);
        }
        if (tet_materials.size() < 2) {
            continue;
        }

        std::array<double, 4> min_material_values;
        for (auto [vid, v] : ranges::zip_view{tet.vertices, min_material_values}) {
            auto& vals = tet_mesh.vertex(vid).data().property.distances;
            v = ranges::min(tet_materials | views::transform([&vals, vid] (auto m) { return vals[m]; }));
        }
        for (std::size_t m = 0; m < n_materials; m++) {
            if (tet_materials.contains(m)) {
                continue;
            }
            auto count = ranges::count_if(ranges::zip_view{tet.vertices, std::move(min_material_values)},  [&tet_mesh, m] (auto pair) {
                auto [vid, v] = pair;
                return tet_mesh.vertex(vid).data().property.distances[m] > v;
            });

            if (count > 1) {
                tet_materials.insert(m);
            }
        }

        auto mi = base_mi;
    }
}
