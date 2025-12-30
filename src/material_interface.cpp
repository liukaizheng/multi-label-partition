#include "material_interface.hpp"
#include "gpf/ids.hpp"
#include "gpf/mesh.hpp"
#include "gpf/surface_mesh.hpp"
#include "tet_mesh.hpp"

#include <implicit_predicates/common.h>
#include <implicit_predicates/implicit_predicates.h>

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

    implicit_predicates::Orientation compute_vert_orientations(const std::array<double, 4>& material, const gpf::VertexId vid);
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

implicit_predicates::Orientation MaterialInterface::compute_vert_orientations(const std::array<double, 4>& material, const gpf::VertexId vid) {
    constexpr std::array<std::array<std::size_t, 5>, 15> M = {{
        {{4, 0, 1, 2, 3}}, //  0: (0, 0, 0, 0)
        {{3, 1, 2, 3, 0}}, //  1: (1, 0, 0, 0)
        {{3, 0, 2, 3, 0}}, //  2: (0, 1, 0, 0)
        {{2, 2, 3, 0, 0}}, //  3: (1, 1, 0, 0)
        {{3, 0, 1, 3, 0}}, //  4: (0, 0, 1, 0)
        {{2, 1, 3, 0, 0}}, //  5: (1, 0, 1, 0)
        {{2, 0, 3, 0, 0}}, //  6: (0, 1, 1, 0)
        {{1, 3, 0, 0, 0}}, //  7: (1, 1, 1, 0)
        {{3, 0, 1, 2, 0}}, //  8: (0, 0, 0, 1)
        {{2, 1, 2, 0, 0}}, //  9: (1, 0, 0, 1)
        {{2, 0, 2, 0, 0}}, // 10: (0, 1, 0, 1)
        {{1, 2, 0, 0, 0}}, // 11: (1, 1, 0, 1)
        {{2, 0, 1, 0, 0}}, // 12: (0, 0, 1, 1)
        {{1, 1, 0, 0, 0}}, // 13: (1, 0, 1, 1)
        {{1, 0, 0, 0, 0}}, // 14: (0, 1, 1, 1)
    }};

    std::size_t vertex_flags = 0;
    std::vector<std::size_t> material_indices;
    for (const auto i : mesh.vertex(vid).data().property.materials) {
        if (i < 4) {
            vertex_flags |= (1 << i);
        } else {
            material_indices.push_back(i - 4);
        }
    }

    auto compute_sign_0 = [this, &material, &material_indices](const std::size_t i) {
        const auto v0 = material[i];
        const auto v = materials[material_indices[0]][i];
        if (v > v0) {
            return implicit_predicates::Orientation::POSITIVE;
        } else if (v < v0) {
            return implicit_predicates::Orientation::NEGATIVE;
        } else {
            return implicit_predicates::Orientation::ZERO;
        }
    };

    auto compute_sign_1 = [this, &material, &material_indices](const std::size_t i, const std::size_t j) {
        const auto& m1 = materials[material_indices[0]];
        const auto& m2 = materials[material_indices[1]];
        double tm1[2] = {m1[i], m1[j]};
        double tm2[2] = {m2[i], m2[j]};
        double tm[2] = {material[i], material[j]};
        return implicit_predicates::mi_orient1d(tm1, tm2, tm);
    };

    auto compute_sign_2 = [this, &material, &material_indices](const std::size_t i, const std::size_t j, const std::size_t k) {
        const auto& m1 = materials[material_indices[0]];
        const auto& m2 = materials[material_indices[1]];
        const auto& m3 = materials[material_indices[2]];
        double tm1[3] = {m1[i], m1[j], m1[k]};
        double tm2[3] = {m2[i], m2[j], m2[k]};
        double tm3[3] = {m3[i], m3[j], m3[k]};
        double tm[3] = {material[i], material[j], material[k]};
        return implicit_predicates::mi_orient2d(tm1, tm2, tm3, tm);
    };

    auto compute_sign_3 = [this, &material, &material_indices]() {
        const auto& m1 = materials[material_indices[0]];
        const auto& m2 = materials[material_indices[1]];
        const auto& m3 = materials[material_indices[2]];
        const auto& m4 = materials[material_indices[3]];
        return implicit_predicates::mi_orient3d(m1.data(), m2.data(), m3.data(), m4.data(), material.data());
    };

    const auto& shape = M[vertex_flags];
    if (shape[0] == 1) {
        return compute_sign_0(shape[1]);
    } else if (shape[0] == 2) {
        return compute_sign_1(shape[1], shape[2]);
    } else if (shape[0] == 3) {
        return compute_sign_2(shape[1], shape[2], shape[3]);
    } else if (shape[0] == 4) {
        return compute_sign_3();
    }
    return implicit_predicates::Orientation::INVALID;
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
