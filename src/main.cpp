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
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/boost/graph/Euler_operations.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/boost/graph/copy_face_graph.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/make_mesh_3.h>

#include <algorithm>
#include <array>
#include <boost/functional/hash.hpp>

#include <gpf/manifold_mesh.hpp>
#include <gpf/mesh.hpp>
#include <mshio/mshio.h>

#include <CLI/CLI.hpp>
#include <gpf/handles.hpp>
#include <gpf/ids.hpp>
#include <gpf/utils.hpp>

#include "tet_mesh.hpp"
#include "material_interface.hpp"
#include "project_polylines_on_mesh.hpp"

#include <fstream>
#include <igl/readOBJ.h>
#include <format>
#include <print>
#include <ranges>
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
using Vb = CGAL::Triangulation_vertex_base_with_info_2<VI, Kernel>;
using Fb_info = CGAL::Triangulation_face_base_with_info_2<bool, Kernel>;
using Fb = CGAL::Constrained_triangulation_face_base_2<Kernel, Fb_info>;
using Tds = CGAL::Triangulation_data_structure_2<Vb, Fb>;
using CDT = CGAL::Constrained_Delaunay_triangulation_2<Kernel, Tds>;
using Mesh_domain = CGAL::Polyhedral_mesh_domain_with_features_3<Kernel>;
using Polyhedron = Mesh_domain::Polyhedron;

using Triangle = Kernel::Triangle_3;
using Iterator = std::vector<Triangle>::const_iterator;
using Primitive = CGAL::AABB_triangle_primitive_3<Kernel, Iterator>;
using AABB_triangle_traits = CGAL::AABB_traits_3<Kernel, Primitive>;
using Tree = CGAL::AABB_tree<AABB_triangle_traits>;

using Tr = CGAL::Mesh_triangulation_3<Mesh_domain>::type;
using C3t3 = CGAL::Mesh_complex_3_in_triangulation_3<Tr>;
using Mesh_criteria = CGAL::Mesh_criteria_3<Tr>;


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
namespace PMP = CGAL::Polygon_mesh_processing;

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

struct ColorRegionBoundaryPart {
    std::size_t color1;
    std::size_t color2;
    std::vector<std::size_t> indices;
};

auto extract_color_boundaries(
    const std::vector<std::array<double, 3>>& points,
    const std::vector<std::vector<std::size_t>>& faces,
    std::vector<std::vector<gpf::FaceId>> color_face_groups
)  {

    std::vector<std::size_t> face_color_indices(faces.size());
    for (std::size_t i = 0; i < color_face_groups.size(); i++) {
        for (const auto fid : color_face_groups[i]) {
            face_color_indices[fid.idx] = i;
        }
    }

    using Mesh = gpf::ManifoldMesh<gpf::Empty, gpf::Empty, gpf::Empty, gpf::Empty>;
    auto mesh = Mesh::new_in(faces);
    auto hash = [](const std::pair<std::size_t, std::size_t>& key) {
        return boost::hash_value(key);
    };
    std::unordered_map<std::pair<std::size_t, std::size_t>, std::vector<gpf::HalfedgeId>, decltype(hash)> boundary_halfedges_map(color_face_groups.size() * 2, std::move(hash));
    for (auto e : mesh.edges()) {
        auto he = e.halfedge();
        auto he_twin = he.twin();
        auto f1 = he.face().id.idx;
        auto f2 = he_twin.face().id.idx;
        auto color_idx1 = face_color_indices[f1];
        auto color_idx2 = face_color_indices[f2];
        if (color_idx1 != color_idx2) {
            if (color_idx1 < color_idx2) {
                boundary_halfedges_map[std::make_pair(color_idx1, color_idx2)].emplace_back(he.id);
            } else {
                boundary_halfedges_map[std::make_pair(color_idx2, color_idx1)].emplace_back(he_twin.id);
            }
        }
    }

    auto sort_halfedges = [&mesh](std::vector<gpf::HalfedgeId>& halfedges) {
        std::unordered_set<gpf::VertexId> vertex_set(halfedges.size());
        for (const auto hid : halfedges) {
            auto he = mesh.halfedge(hid);
            auto va = he.from();
            auto vb = he.to();
            va.data().halfedge = he.id;
            vertex_set.emplace(vb.id);
        }
        gpf::VertexId start_vid{};
        for (const auto hid : halfedges) {
            auto vid = mesh.halfedge(hid).from().id;
            if (!vertex_set.contains(vid)) {
                start_vid = vid;
                break;
            }
        }
        if (!start_vid.valid()) {
            start_vid = mesh.halfedge(halfedges.front()).to().id;
        }
        auto curr_v = mesh.vertex(start_vid);
        for (std::size_t i = 0; i < halfedges.size(); ++i) {
            auto he = curr_v.halfedge();
            halfedges[i] = he.id;
            curr_v = he.to();
        }
    };

    std::vector<gpf::VertexId> outline_vertices;
    std::vector<std::size_t> vertex_map(mesh.n_vertices_capacity(), gpf::kInvalidIndex);

    std::vector<ColorRegionBoundaryPart> boundary_parts;
    boundary_parts.reserve(boundary_halfedges_map.size());
    for (auto&& [pair, halfedges] : boundary_halfedges_map) {
        sort_halfedges(halfedges);
        for (const auto hid : halfedges) {
            auto he = mesh.halfedge(hid);
            for (const auto vid : {he.from().id, he.to().id}) {
                if (vertex_map[vid.idx] == gpf::kInvalidIndex) {
                    vertex_map[vid.idx] = outline_vertices.size();
                    outline_vertices.emplace_back(vid);
                }
            }
        }

        std::vector<std::size_t> boundary_part;
        boundary_part.reserve(halfedges.size() + 1);
        boundary_part.push_back(vertex_map[mesh.halfedge(halfedges.front()).from().id.idx]);
        for (const auto hid : halfedges) {
            boundary_part.emplace_back(vertex_map[mesh.halfedge(hid).to().id.idx]);
        }
        boundary_parts.emplace_back(ColorRegionBoundaryPart{
            .color1 = pair.first,
            .color2 = pair.second,
            .indices = std::move(boundary_part)
        });
    }

    return std::make_tuple(
        outline_vertices | views::transform([&points] (const auto vid) { return points[vid.idx]; }) | ranges::to<std::vector>(),
        std::move(boundary_parts)
    );
}

namespace tet_mesh_boundary {
struct VertexProp {
    std::array<double, 3> pt;
};

struct EdgeProp {
    double len;
};

using Mesh = gpf::ManifoldMesh<VertexProp, gpf::Empty, EdgeProp, gpf::Empty>;

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
    face_tets.reserve(tet_indices.size() * 2);
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
                face_tets[iter.first->second][1] = i;
            }
            idx += 1;
        }
        tets.emplace_back(tet_mesh::Tet{.vertices{gpf::VertexId{t[0]}, gpf::VertexId{t[1]}, gpf::VertexId{t[2]}, gpf::VertexId{t[3]}}, .faces{std::move(tet_faces)}});
    }
    const auto boundary_faces = ranges::iota_view{0ul, faces.size()} | views::filter([&face_tets](std::size_t i) {
        return face_tets[i][1] == gpf::kInvalidIndex;
    }) | ranges::to<std::vector>();
    auto [boundary_mesh_vertices, boundary_mesh] = compress_mesh(points, faces, boundary_faces);
    return std::make_tuple(
        std::move(faces),
        std::move(face_tets),
        std::move(tets),
        std::move(boundary_mesh_vertices),
        std::move(boundary_faces),
        std::move(boundary_mesh)
    );
}

auto project_outline_parts(
    const std::vector<std::array<double, 3>>& outline_points,
    const std::vector<ColorRegionBoundaryPart>& outline_parts,
    Mesh& mesh
) {
    auto polylines = outline_parts | views::transform([] (const auto& part) {
        return part.indices;
    }) | ranges::to<std::vector>();
    project_polylines_on_mesh(outline_points, polylines, mesh);
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

auto tetrahedralize_by_cgal(
    const std::vector<std::array<double, 3>>& points,
    const std::vector<std::vector<std::size_t>>& faces
) {
    Mesh mesh;

    PMP::polygon_soup_to_polygon_mesh(points, faces, mesh);
    Polyhedron polyhedron;
    CGAL::copy_face_graph(mesh, polyhedron);
    Mesh_domain domain{polyhedron};
    domain.detect_features(60); // angle in degrees
    Mesh_criteria criteria(
        CGAL::parameters::edge_size = 1.1,
        CGAL::parameters::facet_angle = 30,
        // CGAL::parameters::facet_size = 0.1,
        // CGAL::parameters::facet_distance = 0.01,
        // CGAL::parameters::cell_radius_edge_ratio = 3,
        CGAL::parameters::cell_size = 1.0
    );
    std::vector<std::array<double, 3>> tet_points;
    std::vector<std::array<std::size_t, 4>> tets;
    C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria);
    std::unordered_map<C3t3::Triangulation::Vertex_handle, std::size_t> vertex_index_map;
    for (auto& v : c3t3.triangulation().finite_vertex_handles()) {
        vertex_index_map[v] = tet_points.size();
        auto& p = v->point();
        tet_points.emplace_back(std::array<double, 3>{{p[0], p[1], p[2]}});
    }
    for (auto cell = c3t3.cells_begin(); cell != c3t3.cells_end(); cell++) {
        std::array<std::size_t, 4> indices;
        for (std::size_t i = 0; i < 4; i++) {
            indices[i] = vertex_index_map[cell->vertex(i)];
        }
        tets.emplace_back(std::move(indices));

    }

    return std::make_pair(std::move(tet_points), std::move(tets));
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
    const std::vector<std::array<double, 3>>& points,
    const std::vector<std::vector<std::size_t>>& faces,
    const std::vector<std::vector<gpf::FaceId>>& label_face_groups
) {
    std::vector<std::vector<Triangle>> triangle_groups;
    triangle_groups.reserve(label_face_groups.size());
    for (const auto& label_faces : label_face_groups) {
        std::vector<Triangle> triangles;
        triangles.reserve(faces.size());
        for (const auto fid : label_faces) {
            auto pts = faces[fid.idx] | views::transform([&points] (auto vid) {
                auto& p = points[vid];
                return Point_3(p[0], p[1], p[2]);
            }) | ranges::to<std::vector>();
            triangles.emplace_back(std::move(pts[0]), std::move(pts[1]), std::move(pts[2]));
        }
        triangle_groups.emplace_back(std::move(triangles));
    }
    return triangle_groups;
}


auto build_tet_mesh(
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
    face_tets.reserve(tet_indices.size() * 2);
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
        std:size_t idx = 0;
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
        f.data().property.cells = std::move(face_tets[f.id.idx]);
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
    Point_3 p{0.0, 1.0, 2.0};
    CLI::App app { "multi-label-partition" };
    std::string mesh_path;
    std::string msh_path;
    app.add_option("-m,--mesh", mesh_path, "Path to mesh file")->required();
    app.add_option("-t,--msh", msh_path, "Path to msh file")->required();
    CLI11_PARSE(app, argc, argv);

    std::vector<std::array<std::size_t, 3>> colors;
    auto [points, faces, label_face_groups] = read_colored_mesh(mesh_path);

    auto [outline_points, outline_parts] = extract_color_boundaries(points, faces, label_face_groups);
    auto [tet_points, tet_indices] = read_msh(msh_path);
    auto [tet_faces, face_tets, tets_temp, boundary_vertices, boundary_faces, boundary_mesh] = tet_mesh_boundary::extract_boundary_from_tets(tet_points, tet_indices);

    auto [tet_mesh, tets] = build_tet_mesh(tet_points, tet_indices);
    auto triangle_groups = build_triangle_groups(points, faces, label_face_groups);

    write_msh("123.obj", tet_mesh);
    setup_neg_distance(tet_mesh, triangle_groups);
    do_material_interface(tets, tet_mesh);
    return 0;
}
