#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_face_base_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Surface_mesh.h>

#include <algorithm>
#include <array>
#include <deque>

#include <gpf/manifold_mesh.hpp>
#include <gpf/mesh.hpp>
#include <gpf/handles.hpp>
#include <gpf/ids.hpp>
#include <gpf/project_polylines_on_mesh.hpp>

#include <CLI/CLI.hpp>

#include <mshio/mshio.h>

#include <boost/functional/hash.hpp>

#include "tet_mesh.hpp"
#include "material_interface.hpp"

#include <fstream>
#include <format>
#include <print>
#include <queue>
#include <ranges>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_2 = Kernel::Point_2;
using Point_3 = Kernel::Point_3;
using Mesh_2 = CGAL::Surface_mesh<Point_2>;
using Mesh = CGAL::Surface_mesh<Point_3>;
using VI = Mesh::Vertex_index;
using HI = Mesh::Halfedge_index;
using EI = Mesh::Edge_index;
using FI = Mesh::Face_index;

using Triangle = Kernel::Triangle_3;
using Iterator = std::vector<Triangle>::const_iterator;
using Primitive = CGAL::AABB_triangle_primitive_3<Kernel, Iterator>;
using AABB_triangle_traits = CGAL::AABB_traits_3<Kernel, Primitive>;
using Tree = CGAL::AABB_tree<AABB_triangle_traits>;

using VMat = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using VMat2 = Eigen::Matrix<double, Eigen::Dynamic, 2>;
using VMat4 = Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>;
using VMat6 = Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>;
using MatXu = Eigen::Matrix<std::size_t, Eigen::Dynamic, Eigen::Dynamic>;
using FMat = Eigen::Matrix<std::size_t, Eigen::Dynamic, 3, Eigen::RowMajor>;
using TMat = Eigen::Matrix<std::size_t, Eigen::Dynamic, 4, Eigen::RowMajor>;
using EMat = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using IVec = Eigen::Matrix<std::size_t, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using RowSpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;

namespace ranges = std::ranges;
namespace views = std::views;

namespace { // unnamed namespace to prevent external linkage
void write_msh(const std::string& name, const tet_mesh::TetMesh& mesh) {
    std::ofstream out(name);
    for (const auto v : mesh.vertices()) {
        const auto& pt = v.data().property.pt;
        std::println(out, "v {} {} {}", pt[0], pt[1], pt[2]);
    }
    for (const auto f : mesh.faces()) {
        std::print(out, "f");
        for (const auto he : f.halfedges()) {
            std::print(out, " {}", he.to().id.idx + 1);
        }
        std::println(out);
    }
}

void write_tet_msh(
    const std::string& filename,
    const std::vector<std::array<double, 3>>& points,
    const std::vector<std::array<std::size_t, 4>>& tets
) {
    mshio::MshSpec spec;
    spec.mesh_format.version = "2.2";
    spec.mesh_format.file_type = 0; // ASCII
    spec.mesh_format.data_size = sizeof(std::size_t);

    // Nodes
    spec.nodes.num_entity_blocks = 1;
    spec.nodes.num_nodes = points.size();
    spec.nodes.min_node_tag = 1;
    spec.nodes.max_node_tag = points.size();

    mshio::NodeBlock node_block;
    node_block.entity_dim = 3;
    node_block.entity_tag = 1;
    node_block.parametric = 0;
    node_block.num_nodes_in_block = points.size();
    node_block.tags.resize(points.size());
    node_block.data.resize(points.size() * 3);
    for (std::size_t i = 0; i < points.size(); ++i) {
        node_block.tags[i] = i + 1;
        node_block.data[i * 3 + 0] = points[i][0];
        node_block.data[i * 3 + 1] = points[i][1];
        node_block.data[i * 3 + 2] = points[i][2];
    }
    spec.nodes.entity_blocks.push_back(std::move(node_block));

    // Elements (tetrahedra, type 4)
    spec.elements.num_entity_blocks = 1;
    spec.elements.num_elements = tets.size();
    spec.elements.min_element_tag = 1;
    spec.elements.max_element_tag = tets.size();

    mshio::ElementBlock elem_block;
    elem_block.entity_dim = 3;
    elem_block.entity_tag = 1;
    elem_block.element_type = 4;
    elem_block.num_elements_in_block = tets.size();
    elem_block.data.resize(tets.size() * 5);
    for (std::size_t i = 0; i < tets.size(); ++i) {
        std::size_t offset = i * 5;
        elem_block.data[offset + 0] = i + 1;
        elem_block.data[offset + 1] = tets[i][0] + 1;
        elem_block.data[offset + 2] = tets[i][1] + 1;
        elem_block.data[offset + 3] = tets[i][2] + 1;
        elem_block.data[offset + 4] = tets[i][3] + 1;
    }
    spec.elements.entity_blocks.push_back(std::move(elem_block));

    mshio::VolumeEntity vol;
    vol.tag = 1;
    spec.entities.volumes.push_back(std::move(vol));

    mshio::save_msh(filename, spec);
}

template<typename Mesh>
void write_mesh(const std::string& name, const Mesh& mesh) {
    std::ofstream out(name);
    for (const auto v : mesh.vertices()) {
        const auto& pt = v.prop().pt;
        std::println(out, "v {} {} {}", pt[0], pt[1], pt[2]);
    }
    for (const auto f : mesh.faces()) {
        std::print(out, "f");
        for (const auto he : f.halfedges()) {
            std::print(out, " {}", he.to().id.idx + 1);
        }
        std::println(out);
    }
}

template<typename Mesh>
void write_faces(const std::string& name, const Mesh& mesh, const std::span<const gpf::FaceId> faces) {
    std::ofstream out(name);
    for (const auto v : mesh.vertices()) {
        const auto& pt = v.prop().pt;
        std::println(out, "v {} {} {}", pt[0], pt[1], pt[2]);
    }
    for (const auto fid : faces) {
        std::print(out, "f");
        auto f = mesh.face(fid);
        for (const auto he : f.halfedges()) {
            std::print(out, " {}", he.to().id.idx + 1);
        }
        std::println(out);
    }
}

template<typename Mesh>
void write_faces(const std::string& name, const Mesh& mesh, const std::vector<std::array<double, 3>>& points, const std::span<const gpf::FaceId> faces) {
    std::ofstream out(name);
    for (const auto& pt : points) {
        std::println(out, "v {} {} {}", pt[0], pt[1], pt[2]);
    }
    for (const auto fid : faces) {
        std::print(out, "f");
        auto f = mesh.face(fid);
        for (const auto he : f.halfedges()) {
            std::print(out, " {}", he.to().id.idx + 1);
        }
        std::println(out);
    }
}

struct RegionBoundaryPart {
    std::size_t region1;
    std::size_t region2;
    std::vector<std::size_t> indices;
};

auto extract_color_boundaries(
    const std::vector<std::array<double, 3>>& points,
    const std::vector<std::vector<std::size_t>>& faces,
    std::vector<std::vector<gpf::FaceId>> color_face_groups
)  {

    struct EdgeProp {
        bool is_boundary = false;
    };
    struct FaceProp {
        std::size_t color_index;
        std::size_t region_index = gpf::kInvalidIndex;
    };

    using Mesh = gpf::ManifoldMesh<gpf::Empty, gpf::Empty, EdgeProp, FaceProp>;
    auto mesh = Mesh::new_in(faces);
    for (std::size_t i = 0; i < color_face_groups.size(); i++) {
        for (const auto fid : color_face_groups[i]) {
            mesh.face_prop(fid).color_index = i;
        }
    }
    for (auto edge : mesh.edges()) {
        auto ha = edge.halfedge();
        auto fa = ha.face();
        auto fb = ha.twin().face();
        if (fa.prop().color_index != fb.prop().color_index) {
            edge.prop().is_boundary = true;
        }
    }

    std::vector<std::size_t> region_colors;
    for (auto face : mesh.faces()) {
        if (face.prop().region_index != gpf::kInvalidIndex) {
            continue;
        }
        const auto region_index = region_colors.size();
        face.prop().region_index = region_index;
        region_colors.push_back(face.prop().color_index);
        std::deque<gpf::FaceId> queue{face.id};
        while (!queue.empty()) {
            auto curr_fid = queue.front();
            queue.pop_front();
            for (auto he: mesh.face(curr_fid).halfedges()) {
                if (he.edge().prop().is_boundary) {
                    continue;
                }
                auto adj_face = he.twin().face();
                auto& adj_face_prop = adj_face.prop();
                if (adj_face_prop.region_index != gpf::kInvalidIndex) {
                    continue;
                }
                adj_face_prop.region_index = region_index;
                queue.push_back(adj_face.id);
            }
        }
    }

    for (std::size_t i = 0; i < region_colors.size(); i++) {
        auto region_faces = mesh.faces() | views::filter([&](const auto& face) {
            return face.prop().region_index == i;
        }) | views::transform([](const auto& face) {
            return face.id;
        }) | ranges::to<std::vector<gpf::FaceId>>();
        write_faces(std::format("region_{}.obj", i), mesh, points, region_faces);
    }

    auto hash = [](const std::pair<std::size_t, std::size_t>& key) {
        return boost::hash_value(key);
    };
    std::unordered_map<std::pair<std::size_t, std::size_t>, std::vector<gpf::HalfedgeId>, decltype(hash)> boundary_halfedges_map(color_face_groups.size() * 2, std::move(hash));
    for (auto e : mesh.edges()) {
        auto he = e.halfedge();
        auto he_twin = he.twin();
        auto f1 = he.face();
        auto f2 = he_twin.face();
        auto region_idx1 = f1.prop().region_index;
        auto region_idx2 = f2.prop().region_index;
        if (region_idx1 != region_idx2) {
            // because the input mesh is oriented outward, but the boundary mesh of tetrahedron is oriented inward
            // we need to record the reversed halfedges
            if (region_idx1 < region_idx2) {
                boundary_halfedges_map[std::make_pair(region_idx1, region_idx2)].emplace_back(he_twin.id);
            } else {
                boundary_halfedges_map[std::make_pair(region_idx2, region_idx1)].emplace_back(he.id);
            }
        }
    }

    auto propagate_halfedges = [&mesh](const gpf::VertexId start_vid, const std::unordered_set<gpf::VertexId>& vertex_set) {
        gpf::VertexId prev_vid{};
        auto curr_v = mesh.vertex(start_vid);
        std::vector<gpf::HalfedgeId> result {};
        while (true) {
            auto he = curr_v.halfedge();
            auto next_v = he.to();
            if (next_v.id == prev_vid || next_v.id == start_vid || !vertex_set.contains(next_v.id)) {
                break;
            } else {
                result.emplace_back(he.id);
                prev_vid = curr_v.id;
                curr_v = next_v;
            }
        }
        return result;
    };

    auto sort_halfedges = [&propagate_halfedges, &mesh](std::vector<gpf::HalfedgeId>& halfedges) {
        std::unordered_set<gpf::VertexId> vertex_set(halfedges.size());
        for (const auto hid : halfedges) {
            auto he = mesh.halfedge(hid);
            auto va = he.from();
            auto vb = he.to();
            va.data().halfedge = he.id;
            vertex_set.emplace(vb.id);
        }
        std::vector<gpf::VertexId> start_vertices{};
        for (const auto hid : halfedges) {
            auto vid = mesh.halfedge(hid).from().id;
            if (!vertex_set.contains(vid)) {
                start_vertices.emplace_back(vid);
            }
        }
        if (start_vertices.empty()) {
            start_vertices.push_back(mesh.halfedge(halfedges.front()).to().id);
        }

        return start_vertices |views::transform([&propagate_halfedges, &vertex_set](gpf::VertexId vid) {
            return propagate_halfedges(vid, vertex_set);
        }) | ranges::to<std::vector>();
    };
    std::vector<gpf::VertexId> outline_vertices;
    std::vector<std::size_t> vertex_map(mesh.n_vertices_capacity(), gpf::kInvalidIndex);

    std::vector<RegionBoundaryPart> boundary_parts;
    boundary_parts.reserve(boundary_halfedges_map.size());
    for (auto&& [pair, all_halfedges] : boundary_halfedges_map) {
        for (const auto hid : all_halfedges) {
            auto he = mesh.halfedge(hid);
            for (const auto vid : {he.from().id, he.to().id}) {
                if (vertex_map[vid.idx] == gpf::kInvalidIndex) {
                    vertex_map[vid.idx] = outline_vertices.size();
                    outline_vertices.emplace_back(vid);
                }
            }
        }
        for (auto&& halfedges: sort_halfedges(all_halfedges)) {
            std::vector<std::size_t> boundary_part;
            boundary_part.reserve(halfedges.size() + 1);
            boundary_part.push_back(vertex_map[mesh.halfedge(halfedges.front()).from().id.idx]);
            for (const auto hid : halfedges) {
                boundary_part.emplace_back(vertex_map[mesh.halfedge(hid).to().id.idx]);
            }
            boundary_parts.emplace_back(RegionBoundaryPart{
                .region1 = pair.first,
                .region2 = pair.second,
                .indices = std::move(boundary_part)
            });
        }
    }

    return std::make_tuple(
        outline_vertices | views::transform([&points] (const auto vid) { return points[vid.idx]; }) | ranges::to<std::vector>(),
        std::move(boundary_parts),
        std::move(region_colors)
    );
}

namespace tet_mesh_boundary {
struct VertexProp {
    std::array<double, 3> pt;
};

struct FaceProp {
    std::size_t region_index = gpf::kInvalidIndex;
};

using Mesh = gpf::ManifoldMesh<VertexProp, gpf::Empty, gpf::Empty, FaceProp>;

auto compress_mesh(
    const std::vector<std::array<double, 3>>& points,
    const std::vector<std::array<std::size_t, 3>>& faces,
    const std::vector<std::size_t>& face_indices
) {
    std::vector<std::size_t> point_map(points.size(), gpf::kInvalidIndex);
    std::vector<std::size_t> mesh_vertices;
    std::vector<std::array<std::size_t, 3>> mesh_faces;
    for (const auto fid : face_indices) {
        auto face = faces[fid];
        for (auto& vid : face) {
            if (point_map[vid] == gpf::kInvalidIndex) {
                point_map[vid] = mesh_vertices.size();
                mesh_vertices.emplace_back(vid);
            }
            vid = point_map[vid];
        }
        mesh_faces.emplace_back(std::move(face));
    }
    auto mesh = Mesh::new_in(mesh_faces);
    for (auto v : mesh.vertices()) {
        v.prop().pt = points[mesh_vertices[v.id.idx]];
    }
    return std::make_tuple(std::move(mesh_vertices), std::move(mesh));
}

auto extract_boundary_from_tets(
    const std::vector<std::array<double, 3>>& points,
    const std::vector<std::array<std::size_t, 4>>& tet_indices
) {
    auto hash = [](const std::array<std::size_t, 3>& key) {
        return boost::hash_value(key);
    };
    auto hash_key = [](const std::array<std::size_t, 3>& key) {
        auto ret = key;
        std::sort(ret.begin(), ret.end());
        return ret;
    };
    std::unordered_map<std::array<std::size_t, 3>, std::size_t, decltype(hash)> face_index_map(tet_indices.size() * 2, std::move(hash));
    std::vector<std::array<std::size_t, 3>> faces;
    std::vector<std::array<std::size_t, 2>> face_tets;
    std::vector<std::array<std::size_t, 4>> tet_faces;
    face_tets.reserve(tet_indices.size() * 2);
    for (std::size_t i = 0; i < tet_indices.size(); ++i) {
        const auto& t = tet_indices[i];
        std::size_t idx = 0;
        std::array<std::size_t, 4> tet_face{};
        for (auto&& face : {
            std::array<std::size_t, 3>{{t[2], t[1], t[3]}},
            std::array<std::size_t, 3>{{t[0], t[2], t[3]}},
            std::array<std::size_t, 3>{{t[1], t[0], t[3]}},
            std::array<std::size_t, 3>{{t[0], t[1], t[2]}}
        }) {
            auto iter = face_index_map.emplace(hash_key(face), faces.size());
            tet_face[idx] = iter.first->second;
            if (iter.second) {
                faces.emplace_back(std::move(face));
                face_tets.emplace_back(std::array<std::size_t, 2>{{i, gpf::kInvalidIndex}});
            } else {
                face_tets[iter.first->second][1] = i;
            }
            idx += 1;
        }
        tet_faces.emplace_back(std::move(tet_face));
    }
    const auto boundary_faces = ranges::iota_view{0ul, faces.size()} | views::filter([&face_tets](std::size_t i) {
        return face_tets[i][1] == gpf::kInvalidIndex;
    }) | ranges::to<std::vector>();
    auto [boundary_mesh_vertices, boundary_mesh] = compress_mesh(points, faces, boundary_faces);

    // For each boundary edge, find all tets sharing it by intersecting
    // the face adjacency lists of its two tet-vertex endpoints
    std::unordered_map<gpf::EdgeId, std::vector<std::size_t>> boundary_edge_tets(boundary_mesh.n_edges());
    auto add_tets_into_map = [&](const gpf::EdgeId eid) {
        auto ha = boundary_mesh.edge(eid).halfedge();
        auto hb = ha.twin();
        auto va = boundary_mesh_vertices[ha.to().id.idx];
        auto vb = boundary_mesh_vertices[hb.to().id.idx];
        auto start_fid = boundary_faces[ha.face().id.idx];
        auto end_fid = boundary_faces[hb.face().id.idx];
        auto start_tid = face_tets[start_fid][0];
        auto end_tid = face_tets[end_fid][0];
        if (start_tid == end_tid) {
            return;
        }

        auto oppo_vertex = [&faces, va, vb](const std::size_t fid) {
            for (const auto vid : faces[fid]) {
                if (vid != va && vid != vb) {
                    return vid;
                }
            }
            return gpf::kInvalidIndex;
        };
        auto curr_fid = start_fid;
        auto curr_tid = start_tid;
        auto vc = oppo_vertex(curr_fid);
        while (true) {
            std::size_t vc_idx{gpf::kInvalidIndex};
            std::size_t vd_idx{gpf::kInvalidIndex};
            for (std::size_t i = 0; i < 4ul; i++) {
                auto vid = tet_indices[curr_tid][i];
                if (vid == vc) {
                    vc_idx = i;
                    if (vd_idx != gpf::kInvalidIndex) {
                        break;
                    }
                } else if (vid != va && vid != vb) {
                    vd_idx = i;
                    if (vc_idx != gpf::kInvalidIndex) {
                        break;
                    }
                }
            }
            auto vd = tet_indices[curr_tid][vd_idx];
            curr_fid = tet_faces[curr_tid][vc_idx];
            curr_tid = face_tets[curr_fid][0] == curr_tid ? face_tets[curr_fid][1] : face_tets[curr_fid][0];
            if (curr_tid == end_tid) {
                break;
            }
            boundary_edge_tets[eid].push_back(curr_tid);
            vc = vd;
        }
    };
    for (auto e : boundary_mesh.edges()) {
        auto [va, vb] = boundary_mesh.e_vertices(e.id);
        auto v1 = boundary_mesh_vertices[va.idx];
        auto v2 = boundary_mesh_vertices[vb.idx];
        if (v1 > v2) {
            std::swap(v1, v2);
        }
        if (v1 == 1234 && v2 == 25944) {
            const auto a = 2;
        }
        add_tets_into_map(e.id);
    }

    return std::make_tuple(
        std::move(faces),
        std::move(face_tets),
        std::move(boundary_mesh_vertices),
        std::move(boundary_faces),
        std::move(boundary_mesh),
        std::move(boundary_edge_tets)
    );
}

[[nodiscard]] auto group_child_edges_by_parent(
    const std::unordered_map<gpf::EdgeId, gpf::EdgeId>& edge_parent_map
) {
    std::unordered_map<gpf::EdgeId, std::vector<gpf::EdgeId>> edge_parent_to_children;
    for (const auto [child_eid, parent_eid] : edge_parent_map) {
        edge_parent_to_children[parent_eid].push_back(child_eid);
    }
    return edge_parent_to_children;
}

auto build_split_edge_chain(Mesh& boundary_mesh, const std::vector<gpf::EdgeId>& child_edges) {
    std::unordered_map<gpf::VertexId, std::vector<gpf::HalfedgeId>> vertex_to_halfedges_map(child_edges.size() + std::size_t{1});
    for (const auto eid : child_edges) {
        for (const auto he : boundary_mesh.edge(eid).halfedges()) {
            vertex_to_halfedges_map[he.from().id].push_back(he.id);
        }
    }

    std::vector<gpf::HalfedgeId> chain;
    chain.reserve(child_edges.size());
    auto start_it = ranges::find_if(vertex_to_halfedges_map, [](const auto& pair) {
        return pair.second.size() == 1;
    });
    assert(start_it != vertex_to_halfedges_map.end());
    auto curr_hid = start_it->second[0];
    while (true) {
        chain.emplace_back(curr_hid);
        auto vid = boundary_mesh.he_to(curr_hid);
        auto& candidates = vertex_to_halfedges_map[vid];
        if (candidates.size() != 2) {
            break;
        }
        curr_hid = boundary_mesh.he_edge(candidates[0]) == boundary_mesh.he_edge(curr_hid) ? candidates[1] : candidates[0];
    }
    assert(chain.size() == child_edges.size());
    return chain;
}

struct SplitEdgeInfo {
    gpf::EdgeId eid;
    std::vector<gpf::HalfedgeId> chain;
    std::vector<std::size_t> interior_tets;
    std::array<std::size_t, 2> vertices;
};

void split_tets_by_face(
    std::vector<std::array<double, 3>>& tet_points,
    const std::vector<std::array<std::size_t, 3>>& tet_faces,
    const std::vector<std::array<std::size_t, 2>>& face_tets,
    std::vector<std::array<std::size_t, 4>>& tets,
    const std::vector<std::size_t>& boundary_faces,
    const std::vector<std::size_t>& boundary_to_tet_vertex,
    const std::vector<bool>& vertex_is_boundary,
    Mesh& boundary_mesh,
    const std::unordered_map<gpf::FaceId, gpf::FaceId>& face_parent_map,
    std::vector<SplitEdgeInfo>& split_edge_info_vec
) {
    auto edge_hash = [](const std::array<std::size_t, 2>& arr) {
        return boost::hash_value(arr);
    };

    std::unordered_map<std::array<std::size_t, 2>, std::size_t, decltype(edge_hash)> split_tet_edge_index_map(split_edge_info_vec.size());
    for (std::size_t i = 0; i < split_edge_info_vec.size(); i++) {
        split_tet_edge_index_map.emplace(split_edge_info_vec[i].vertices, i);
    }

    // Group boundary_mesh faces by their parent
    std::unordered_map<gpf::FaceId, std::vector<gpf::FaceId>> parent_to_children;
    for (const auto [fid, parent] : face_parent_map) {
        parent_to_children[parent].push_back(fid);
    }

    // Pre-compute per-split-face info before any mutation,
    // grouped by tet so each tet is processed exactly once.
    struct SplitFaceInfo {
        std::size_t local_vd_idx; // the opposite vertex for this face
        std::vector<gpf::FaceId> children;
    };
    std::unordered_map<std::size_t, std::vector<SplitFaceInfo>> tet_split_faces;

    for (auto& [parent_fid, children] : parent_to_children) {
        assert(children.size() > 1);

        auto boundary_face_idx = parent_fid.idx;
        auto tet_face_idx = boundary_faces[boundary_face_idx];

        const auto tet_idx = face_tets[tet_face_idx][0];
        assert(tet_idx != gpf::kInvalidIndex);
        const auto& tet = tets[tet_idx];

        // Compute vd from the *unmodified* tets array
        const auto& boundary_face = tet_faces[tet_face_idx];
        const auto local_vd_idx = *ranges::find_if(ranges::iota_view{std::size_t{0}, std::size_t{4}}, [&boundary_face, &tet](const auto idx) {
            return tet[idx] != boundary_face[0] && tet[idx] != boundary_face[1] && tet[idx] != boundary_face[2];
        });

        tet_split_faces[tet_idx].push_back(SplitFaceInfo{local_vd_idx, std::move(children)});
    }

    // Helper: build a sub-tet from a boundary mesh sub-face + an apex vertex
    auto build_sub_tet = [&boundary_mesh, &boundary_to_tet_vertex] (std::array<std::size_t, 4>& tet_indices, const gpf::FaceId fid, std::size_t apex) {
        for(auto&& [idx, he] : views::zip(tet_indices, boundary_mesh.face(fid).halfedges())) {
            idx = boundary_to_tet_vertex[he.to().id.idx];
        }
        tet_indices[3] = apex;
    };

    auto append_tet_for_parent_edges = [&split_edge_info_vec, &split_tet_edge_index_map, &tets, &vertex_is_boundary](const std::size_t tid) {
        constexpr std::array<std::array<std::size_t, 2>, 3> FACE_EDGES = {{
            {0, 1},
            {1, 2},
            {2, 0},
        }};
        const auto& tet = tets[tid];
        if (ranges::any_of(std::span<const std::size_t, 3>{tet.data(), 3}, [&vertex_is_boundary](auto idx) { return !vertex_is_boundary[idx]; })) {
            return;
        }
        for (const auto [local_va, local_vb] : FACE_EDGES) {
            auto va = tet[local_va];
            auto vb = tet[local_vb];
            auto it = split_tet_edge_index_map.find(
                va < vb ? std::array<std::size_t, 2>{va, vb} : std::array<std::size_t, 2>{vb, va}
            );
            if (it != split_tet_edge_index_map.end()) {
                split_edge_info_vec[it->second].interior_tets.push_back(tid);
            }
        }
    };

    for (const auto& [tet_idx, split_faces] : tet_split_faces) {
        if (split_faces.size() == 1) {
            // Single split face: fan from opposite vertex (no Steiner point needed)
            const auto& info = split_faces[0];
            auto& tet = tets[tet_idx];
            const auto vd = tet[info.local_vd_idx];

            build_sub_tet(tet, info.children[0], vd);
            for (std::size_t i = 1; i < info.children.size(); ++i) {
                std::array<std::size_t, 4> new_tet;
                build_sub_tet(new_tet, info.children[i], vd);
                tets.push_back(std::move(new_tet));
            }
        } else {
            // Multiple split faces: insert centroid as Steiner point and fan from it
            // to all boundary sub-triangles AND unsplit faces of this tet.
            const auto orig_tet = tets[tet_idx]; // copy before mutation
            std::array<double, 3> centroid{};
            auto eigen_centroid = Eigen::Vector3d::Map(centroid.data());
            for (std::size_t i = 0; i < 4; i++) {
                eigen_centroid += Eigen::Vector3d::Map(tet_points[orig_tet[i]].data());
            }
            eigen_centroid *= 0.25;

            auto centroid_idx = tet_points.size();
            tet_points.emplace_back(centroid);

            // Identify which of the 4 tet faces are NOT split.
            // Each face of a tet is defined by 3 of its 4 vertices; the face opposite
            // vertex v contains the other 3 vertices. So a split face with vd=v means
            // the face containing {all vertices except v} is split.
            std::array<bool, 4> local_face_is_split{{false, false, false, false}};
            for (const auto& info : split_faces) {
                local_face_is_split[info.local_vd_idx] = true;
            }

            bool first = true;
            // Fan centroid to each split boundary face's sub-triangles
            for (const auto& info : split_faces) {
                for (auto child_fid : info.children) {
                    if (first) {
                        build_sub_tet(tets[tet_idx], child_fid, centroid_idx);
                        first = false;
                    } else {
                        std::array<std::size_t, 4> new_tet;
                        build_sub_tet(new_tet, child_fid, centroid_idx);
                        tets.emplace_back(std::move(new_tet));
                    }
                }
            }

            // Fan centroid to each unsplit face of the tet.
            // The 4 faces of tet [v0,v1,v2,v3] are opposite to v0,v1,v2,v3 respectively.
            // A face is unsplit if its opposite vertex is NOT in local_face_is_split.
            constexpr std::array<std::array<std::size_t, 3>, 4> LOCAL_FACE_VERTICES = {{
                {2, 1, 3}, // face opposite vertex 0
                {0, 2, 3}, // face opposite vertex 1
                {1, 0, 3}, // face opposite vertex 2
                {0, 1, 2}, // face opposite vertex 3
            }};
            for (std::size_t local_v = 0; local_v < 4; local_v++) {
                if (local_face_is_split[local_v]) {
                    continue; // this face is split, already handled above
                }
                const auto& lf = LOCAL_FACE_VERTICES[local_v];
                assert(!first); // at least one split face was processed
                tets.push_back({orig_tet[lf[0]], orig_tet[lf[1]], orig_tet[lf[2]], centroid_idx});
                append_tet_for_parent_edges(tets.size() - 1);
            }
        }
    }
}

void split_tets_by_edge(
    std::vector<std::array<std::size_t, 4>>& tets,
    const std::vector<std::size_t>& boundary_to_tet_vertex,
    Mesh& boundary_mesh,
    const std::vector<SplitEdgeInfo>& split_edge_info_vec
) {

    std::unordered_map<std::size_t, std::vector<std::size_t>> tet_split_edges;
    for (std::size_t i = 0; i < split_edge_info_vec.size(); i++) {
        for (const auto tid : split_edge_info_vec[i].interior_tets) {
            tet_split_edges[tid].push_back(i);
        }
    }

    auto split_tet_by_edge = [&tets, &split_edge_info_vec, &boundary_to_tet_vertex, &boundary_mesh](const std::size_t tid, const std::size_t split_edge_idx) noexcept {
        auto& tet = tets[tid];
        const auto& split_edge_info = split_edge_info_vec[split_edge_idx];
        const auto [e_va, e_vb] = split_edge_info.vertices;
        std::size_t e_va_local = gpf::kInvalidIndex, e_vb_local = gpf::kInvalidIndex;
        for (std::size_t i = 0; i < 4; i++) {
            if (tet[i] == e_va) {
                e_va_local = i;
                if (e_vb_local != gpf::kInvalidIndex) {
                    break;
                }
            } else if (tet[i] == e_vb) {
                e_vb_local = i;
                if (e_va_local != gpf::kInvalidIndex) {
                    break;
                }
            }
        }
        if (e_va_local == gpf::kInvalidIndex || e_vb_local == gpf::kInvalidIndex) {
            return;
        }
        auto hid_local = gpf::oriented_index(std::size_t{5} - tet_mesh::Tet::edge_index(e_va_local, e_vb_local), e_va_local > e_vb_local);
        constexpr std::array<std::array<std::size_t, 2>, 12> EDGE_TO_VC_CD = {{
            {2, 3},
            {3, 2},
            {3, 1},
            {1, 3},
            {1, 2},
            {2, 1},
            {0, 3},
            {3, 0},
            {2, 0},
            {0, 2},
            {0, 1},
            {1, 0}
        }};
        auto [vc, vd] = EDGE_TO_VC_CD[hid_local];
        vc = tet[vc];
        vd = tet[vd];

        const auto& chain = split_edge_info.chain;

        tet = {e_va, boundary_to_tet_vertex[boundary_mesh.he_to(chain.front()).idx], vc, vd};
        for (std::size_t i = 1; i < chain.size(); i++) {
            auto hid = chain[i];
            auto va = boundary_to_tet_vertex[boundary_mesh.he_from(hid).idx];
            auto vb = boundary_to_tet_vertex[boundary_mesh.he_to(hid).idx];
            tets.emplace_back(std::array<std::size_t, 4>{va, vb, vc, vd});
        }
    };

    for (auto&& [tid, edge_indices] : std::move(tet_split_edges)) {
        if (edge_indices.size() == 1) {
            split_tet_by_edge(tid, edge_indices[0]);
        } else {
            // ensure that if two tets share a face with split edges,
            // then they produce identical sub-triangulation on that face
            ranges::sort(edge_indices, [&split_edge_info_vec](auto i, auto j) {
                return ranges::lexicographical_compare(split_edge_info_vec[i].vertices, split_edge_info_vec[j].vertices);
            });
            std::vector<std::size_t> active_tets{tid};
            std::vector<std::size_t> active_tets_swap;
            for (const auto split_edge_idx : edge_indices) {
                for (const auto tid : active_tets) {
                    const auto n_old_tets = tets.size();
                    split_tet_by_edge(tid, split_edge_idx);
                    active_tets_swap.push_back(tid);
                    for (std::size_t i = n_old_tets; i < tets.size(); i++) {
                        active_tets_swap.push_back(i);
                    }
                }
                active_tets.swap(active_tets_swap);
                active_tets_swap.clear();
            }
        }
    }
}

void rebuild_tets_from_split_boundary(
    std::vector<std::array<double, 3>>& tet_points,
    const std::vector<std::array<std::size_t, 3>>& tet_faces,
    const std::vector<std::array<std::size_t, 2>>& face_tets,
    std::vector<std::array<std::size_t, 4>>& tets,
    const std::vector<std::size_t>& boundary_faces,
    std::vector<std::size_t>&& boundary_to_tet_vertex,
    Mesh& boundary_mesh,
    const std::unordered_map<gpf::FaceId, gpf::FaceId>& face_parent_map,
    const std::unordered_map<gpf::EdgeId, gpf::EdgeId>& edge_parent_map,
    const std::unordered_map<gpf::EdgeId, std::vector<std::size_t>>& boundary_edge_tets
) {

    std::vector<bool> is_boundary_vertex(tet_points.size(), false);
    for (const auto idx : boundary_to_tet_vertex) {
        is_boundary_vertex[idx] = true;
    }

    const auto n_old_boundary_vertices = boundary_to_tet_vertex.size();
    boundary_to_tet_vertex.resize(boundary_mesh.n_vertices_capacity(), gpf::kInvalidIndex);
    for (std::size_t i = n_old_boundary_vertices; i < boundary_to_tet_vertex.size(); i++) {
        boundary_to_tet_vertex[i] = tet_points.size();
        tet_points.emplace_back(boundary_mesh.vertex_prop(gpf::VertexId{i}).pt);
    }
    is_boundary_vertex.resize(tet_points.size(), false);
    auto edge_parent_to_children = group_child_edges_by_parent(edge_parent_map);
    auto split_edge_info_vec = std::move(edge_parent_to_children) | views::transform([&boundary_mesh, &boundary_edge_tets, &boundary_to_tet_vertex] (auto&& pair) {
        auto it = boundary_edge_tets.find(pair.first);
        auto info = SplitEdgeInfo {
            .eid{pair.first},
            .chain{build_split_edge_chain(boundary_mesh, pair.second)},
            .interior_tets{it == boundary_edge_tets.end() ? std::vector<std::size_t>{} : it->second }
        };
        info.vertices[0] = boundary_to_tet_vertex[boundary_mesh.he_from(info.chain.front()).idx];
        info.vertices[1] = boundary_to_tet_vertex[boundary_mesh.he_to(info.chain.back()).idx];
        if (info.vertices[0] > info.vertices[1]) {
            std::swap(info.vertices[0], info.vertices[1]);
            ranges::reverse(info.chain);
            ranges::for_each(info.chain, [&boundary_mesh](auto& h) { h = boundary_mesh.he_twin(h); });
        }
        return info;
    }) | ranges::to<std::vector>();

    split_tets_by_face(tet_points, tet_faces, face_tets, tets, boundary_faces, boundary_to_tet_vertex, is_boundary_vertex, boundary_mesh, face_parent_map, split_edge_info_vec);
    split_tets_by_edge(tets, boundary_to_tet_vertex, boundary_mesh, split_edge_info_vec);
}

void mark_face_region(
    const std::vector<std::vector<gpf::HalfedgeId>>& path_halfedges,
    const std::vector<RegionBoundaryPart>& outline_parts,
    Mesh& mesh
) {
    // Seed face colors from boundary halfedges
    // The halfedge orientation: face of halfedge has color1, face of twin has color2
    std::queue<gpf::FaceId> queue;
    for (std::size_t i = 0; i < path_halfedges.size(); ++i) {
        const auto& part = outline_parts[i];
        for (const auto hid : path_halfedges[i]) {
            auto he = mesh.halfedge(hid);
            auto f1 = he.face();
            auto f2 = he.twin().face();
            if (f1.prop().region_index == gpf::kInvalidIndex) {
                f1.prop().region_index = part.region1;
                queue.push(f1.id);
            }
            if (f2.prop().region_index == gpf::kInvalidIndex) {
                f2.prop().region_index = part.region2;
                queue.push(f2.id);
            }
        }
    }

    // Flood fill to propagate colors
    while (!queue.empty()) {
        auto fid = queue.front();
        queue.pop();
        auto f = mesh.face(fid);
        auto region = f.prop().region_index;
        for (auto he : f.halfedges()) {
            auto neighbor = he.twin().face();
            if (neighbor.prop().region_index == gpf::kInvalidIndex) {
                neighbor.prop().region_index = region;
                queue.push(neighbor.id);
            }
        }
    }
}

void project_outline_parts(
    std::vector<std::array<double, 3>>& outline_points,
    const std::vector<RegionBoundaryPart>& outline_parts,
    Mesh& mesh,
    std::unordered_map<gpf::FaceId, gpf::FaceId>& face_parent_map,
    std::unordered_map<gpf::EdgeId, gpf::EdgeId>& edge_parent_map
) {
    auto path_halfedges = gpf::project_polylines_on_mesh<3>(
        outline_points,
        outline_parts | views::transform([] (const auto& part) {
            return part.indices;
        }) | ranges::to<std::vector>(),
        mesh,
        &face_parent_map,
        &edge_parent_map
    );

    mark_face_region(path_halfedges, outline_parts, mesh);
}

}

auto read_colored_mesh(const std::string& file_name) {
    std::ifstream input_file(file_name, std::ios::in);
    int n_v, n_f, n_e;
    std::string line;
    std::vector<std::array<double, 3>> points;
    std::vector<std::vector<std::size_t>> faces;
    getline(input_file, line); // file format
    {
        line.clear();
        getline(input_file, line);
        std::stringstream line_stream;
        line_stream.str(line);
        line_stream >> n_v >> n_f >> n_e;
    }

    std::vector<std::array<std::size_t, 3>> face_colors(n_f);
    std::array<double, 3> pos;
    for(int i = 0; i < n_v; ++i) {
        line.clear();
        getline(input_file, line);
        std::stringstream line_stream;
        line_stream.str(line);
        line_stream >> pos[0] >> pos[1] >> pos[2];
        points.emplace_back(pos);
    }

    std::size_t indices[3];
    std::size_t color[3];
    int f;
    for(int i = 0; i < n_f; ++i) {
        line.clear();
        getline(input_file, line);
        std::stringstream line_stream;
        line_stream.str(line);
        line_stream >> f;
        for(int j = 0; j < f; ++j) {
        line_stream >> indices[j];
        }
        line_stream >> color[0] >> color[1] >> color[2];
        face_colors[i] = {color[0], color[1], color[2]};
        faces.emplace_back(std::vector{indices[0], indices[1], indices[2]});
    }
    input_file.close();
    auto hash = [](const std::array<std::size_t, 3>& key) {
        return boost::hash_value(key);
    };
    std::unordered_map<std::array<std::size_t, 3>, std::vector<std::size_t>, decltype(hash)> color_to_face_map(faces.size(), hash);
    for (std::size_t i = 0; i < faces.size(); ++i) {
        color_to_face_map[face_colors[i]].push_back(i);
    }
    return std::make_tuple(
        std::move(points),
        std::move(faces),
        std::move(color_to_face_map)
            | std::views::elements<1>
            | views::transform([](auto&& faces) {
                return faces | views::transform([](const auto fid) {
                    return gpf::FaceId{fid}; }) | ranges::to<std::vector>();
                })
            | ranges::to<std::vector>()
    );
}

auto read_msh(const std::string& file_name) {
    mshio::MshSpec spec = mshio::load_msh(file_name);
    // Build a map from node tag to index
    std::unordered_map<std::size_t, std::size_t> tag_to_index;
    std::vector<std::array<double, 3>> points;
    for (const auto& block : spec.nodes.entity_blocks) {
        for (std::size_t i = 0; i < block.num_nodes_in_block; ++i) {
            std::size_t tag = block.tags[i];
            tag_to_index[tag] = points.size();
            points.push_back({
                block.data[i * 3 + 0],
                block.data[i * 3 + 1],
                block.data[i * 3 + 2]
            });
        }
    }
    std::vector<std::array<std::size_t, 4>> tets;
    for (const auto& block : spec.elements.entity_blocks) {
        // Element type 4 is tetrahedron in Gmsh
        if (block.element_type == 4) {
            std::size_t nodes_per_elem = mshio::nodes_per_element(block.element_type);
            for (std::size_t i = 0; i < block.num_elements_in_block; ++i) {
                // data layout: [elem_tag, node1, node2, node3, node4, ...]
                std::size_t offset = i * (nodes_per_elem + 1);
                tets.push_back({
                    tag_to_index[block.data[offset + 1]],
                    tag_to_index[block.data[offset + 2]],
                    tag_to_index[block.data[offset + 3]],
                    tag_to_index[block.data[offset + 4]]
                });
            }
        }
    }
    return std::make_pair(std::move(points), std::move(tets));
}

auto build_triangle_groups(
    const tet_mesh_boundary::Mesh& mesh,
    const std::size_t n_materials
) {
    std::vector<std::vector<Triangle>> triangle_groups(n_materials);
    for (const auto face : mesh.faces()) {
        const auto region_idx = face.prop().region_index;
            auto pts = face.halfedges() | views::transform([](auto he) {
                const auto& pt = he.to().prop().pt;
                return Point_3(pt[0], pt[1], pt[2]);
            }) | ranges::to<std::vector>();
        triangle_groups[region_idx].emplace_back(std::move(pts[0]), std::move(pts[1]), std::move(pts[2]));
    }
    return triangle_groups;
}


auto build_tet_mesh(
    const std::vector<std::array<double, 3>>& points,
    const std::vector<std::array<std::size_t, 4>>& tet_indices,
    std::vector<std::array<std::size_t, 3>>& faces,
    std::vector<std::array<std::size_t, 2>>& face_tets
) {
    faces.clear();
    face_tets.clear();
    auto hash = [](const std::array<std::size_t, 3>& key) {
        return boost::hash_value(key);
    };
    auto hash_key = [](const std::array<std::size_t, 3>& key) {
        auto ret = key;
        std::sort(ret.begin(), ret.end());
        return ret;
    };
    std::unordered_map<std::array<std::size_t, 3>, std::size_t, decltype(hash)> face_index_map(tet_indices.size() * 2, std::move(hash));
    std::vector<tet_mesh::Tet> tets;
    for (std::size_t i = 0; i < tet_indices.size(); ++i) {
        const auto& t = tet_indices[i];
        std::array<gpf::FaceId, 4> tet_faces;
        std::size_t idx = 0;
        for (auto&& face : {
            std::array<std::size_t, 3>{{t[2], t[1], t[3]}},
            std::array<std::size_t, 3>{{t[0], t[2], t[3]}},
            std::array<std::size_t, 3>{{t[1], t[0], t[3]}},
            std::array<std::size_t, 3>{{t[0], t[1], t[2]}}
        }) {
            auto iter = face_index_map.emplace(hash_key(face), faces.size());
            tet_faces[idx] = gpf::FaceId{iter.first->second};
            if (iter.second) {
                faces.emplace_back(std::move(face));
                face_tets.emplace_back(std::array<std::size_t, 2>{{i, gpf::kInvalidIndex}});
            } else {
                assert(face_tets[iter.first->second][1] == gpf::kInvalidIndex);
                face_tets[iter.first->second][1] = i;
            }
            idx += 1;
        }
        tets.emplace_back(tet_mesh::Tet{.vertices{gpf::VertexId{t[0]}, gpf::VertexId{t[1]}, gpf::VertexId{t[2]}, gpf::VertexId{t[3]}}, .faces{std::move(tet_faces)}});
    }
    auto mesh = tet_mesh::TetMesh::new_in(std::move(faces));
    for (auto& tet : tets) {
        const auto& vertices = tet.vertices;
        auto& edges = tet.edges;
        std::size_t idx = 0;
        for (std::size_t i = 0; i < 3; i++) {
            for (std::size_t j = i + 1; j < 4; j++) {
                const auto eid = mesh.e_from_vertices(vertices[i], vertices[j]);
                edges[idx++] = eid;
            }
        }
    }
    for (auto v : mesh.vertices()) {
        v.data().property.pt = points[v.id.idx];
    }

    for (auto f : mesh.faces()) {
        auto& f_prop = f.prop();
        f_prop.cells = std::move(face_tets[f.id.idx]);
        if (f_prop.cells[1] == gpf::kInvalidIndex) {
            for (auto he : f.halfedges()) {
                he.to().prop().on_boundary = true;
            }
        }
    }

    return std::make_pair(std::move(mesh), std::move(tets));
}

auto setup_neg_distance(tet_mesh::TetMesh& mesh, const std::vector<std::vector<Triangle>>& triangle_groups) {
    std::vector<Tree> trees;
    trees.reserve(triangle_groups.size());
    for (const auto& triangles : triangle_groups) {
        trees.emplace_back(triangles.begin(), triangles.end());
        trees.back().accelerate_distance_queries();
    }
    for (auto v : mesh.vertices()) {
        auto& dists = v.data().property.distances;
        const auto& p = v.data().property.pt;
        dists.resize(trees.size());
        Point_3 pt(p[0], p[1], p[2]);
        for (auto [d, tree] : ranges::zip_view(dists, trees)) {
            // auto [closest_point, primitive] = tree.closest_point_and_primitive(pt);
            // auto fid = primitive - triangle_groups[idx++].begin();
            d = -std::sqrt(tree.squared_distance(pt));
        }
    }
}
}

int main(int argc, char* argv[]) {
    CLI::App app { "multi-label-partition" };
    std::string mesh_path;
    std::string msh_path;
    app.add_option("-m,--mesh", mesh_path, "Path to mesh file")->required();
    app.add_option("-t,--msh", msh_path, "Path to msh file")->required();
    CLI11_PARSE(app, argc, argv);

    std::vector<std::array<std::size_t, 3>> colors;
    auto [points, faces, label_face_groups] = read_colored_mesh(mesh_path);
    auto [tet_points, tet_indices] = read_msh(msh_path);

    auto [outline_points, outline_parts, region_colors] = extract_color_boundaries(points, faces, label_face_groups);
    auto [tet_faces, face_tets, boundary_vertices, boundary_faces, boundary_mesh, boundary_edge_tets] = tet_mesh_boundary::extract_boundary_from_tets(tet_points, tet_indices);
    std::unordered_map<gpf::FaceId, gpf::FaceId> face_parent_map;
    std::unordered_map<gpf::EdgeId, gpf::EdgeId> edge_parent_map;
    tet_mesh_boundary::project_outline_parts(outline_points, outline_parts, boundary_mesh, face_parent_map, edge_parent_map);
    {
        for (std::size_t i = 0; i < region_colors.size(); ++i) {
            auto region_faces = boundary_mesh.faces() | views::filter([i] (auto f) { return f.prop().region_index == i;}) |
                views::transform([](auto f) { return f.id; }) |
                ranges::to<std::vector>();
                write_faces(std::format("region_{}.obj", i), boundary_mesh, region_faces);
        }
        write_mesh("fine.obj", boundary_mesh);
    }

    // Rebuild tets from split boundary faces and edges
    tet_mesh_boundary::rebuild_tets_from_split_boundary(
        tet_points, tet_faces, face_tets, tet_indices, boundary_faces, std::move(boundary_vertices), boundary_mesh, face_parent_map, edge_parent_map, boundary_edge_tets
    );

    auto [tet_mesh, tets] = build_tet_mesh(tet_points, tet_indices, tet_faces, face_tets);
    {
        auto boundary_faces = tet_mesh.faces() | views::filter([](auto f) { return f.prop().cells[1] == gpf::kInvalidIndex; }) |
            views::transform([](auto f) { return f.id; }) |
            ranges::to<std::vector>();
        auto fid = boundary_faces[10033];
        auto face_vertices = tet_mesh.face(fid).halfedges() | views::transform([](auto he) { return he.to().id.idx; }) |
            ranges::to<std::vector>();
        write_faces("boundary.obj", tet_mesh, boundary_faces);
    }
    auto triangle_groups = build_triangle_groups(boundary_mesh, region_colors.size());
    write_msh("123.obj", tet_mesh);
    write_tet_msh("output.msh", tet_points, tet_indices);

    setup_neg_distance(tet_mesh, triangle_groups);
    do_material_interface(tets, tet_mesh);
    return 0;
}
