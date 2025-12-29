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

#include <array>
#include <boost/functional/hash.hpp>

#include <CLI/CLI.hpp>
#include "tet_mesh.hpp"

#include <fstream>
#include <igl/readOBJ.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <format>
#include <print>
#include <ranges>
#include <unordered_map>
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

auto view_ts = std::views::transform;

void write_msh(const std::string& name, const VMat& V, const TMat& T) {
    std::ofstream out(name);
    for (Eigen::Index i = 0; i < V.rows(); ++i) {
        std::println(out, "v {} {} {}", V(i, 0), V(i, 1), V(i, 2));
    }
    for (Eigen::Index i = 0; i < T.rows(); ++i) {
        const auto t = T.row(i);
        std::println(out, "f {} {} {}", t(0) + 1, t(1) + 1, t(2) + 1);
        std::println(out, "f {} {} {}", t(0) + 1, t(2) + 1, t(3) + 1);
        std::println(out, "f {} {} {}", t(0) + 1, t(3) + 1, t(1) + 1);
        std::println(out, "f {} {} {}", t(1) + 1, t(3) + 1, t(2) + 1);
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
            | view_ts([](auto&& faces) {
                return faces | view_ts([](const auto fid) {
                    return gpf::FaceId{fid}; }) | std::ranges::to<std::vector>();
                })
            | std::ranges::to<std::vector>()
    );
}


auto build_tet_mesh(
    const VMat& TV,
    const TMat& TT,
    const FMat& TF
) {
    auto hash = [](const std::array<std::size_t, 3>& key) {
        return boost::hash_value(key);
    };
    auto hash_key = [](const std::array<std::size_t, 3>& key) {
        auto ret = key;
        std::sort(ret.begin(), ret.end());
        return ret;
    };
    std::unordered_map<std::array<std::size_t, 3>, std::size_t, decltype(hash)> face_index_map(TT.rows() * 2, std::move(hash));
    std::vector<std::array<std::size_t, 3>> faces;
    for (std::size_t i = 0; i < TF.rows(); ++i) {
        std::array<std::size_t, 3> face = {TF(i, 0), TF(i, 1), TF(i, 2)};
        face_index_map.emplace(hash_key(face), faces.size());
        faces.emplace_back(std::move(face));
    }

    std::vector<std::array<std::size_t, 2>> face_tets(faces.size(), {{gpf::kInvalidIndex, gpf::kInvalidIndex}});
    std::vector<std::size_t> signed_tet_faces;
    for (std::size_t i = 0; i < TT.rows(); ++i) {
        auto t = TT.row(i);
        for (auto&& face : {
            std::array<std::size_t, 3>{{t[2], t[1], t[3]}},
            std::array<std::size_t, 3>{{t[0], t[2], t[3]}},
            std::array<std::size_t, 3>{{t[1], t[0], t[3]}},
            std::array<std::size_t, 3>{{t[0], t[1], t[2]}}
        }) {
            auto iter = face_index_map.emplace(hash_key(face), faces.size());
            if (iter.second) {
                faces.emplace_back(std::move(face));
                face_tets.emplace_back(std::array<std::size_t, 2>{{i, gpf::kInvalidIndex}});
            } else {
                if (face_tets[iter.first->second][0] == gpf::kInvalidIndex) {
                    face_tets[iter.first->second][0] = i;
                } else {
                    face_tets[iter.first->second][1] = i;
                }
            }
        }
    }
    auto mesh = tet_mesh::TetMesh::new_in(std::move(faces));
    for (auto v : mesh.vertices()) {
        auto& p = v.data().property.pt;
        auto q = TV.row(v.id.idx).eval();
        p[0] = q[0];
        p[1] = q[1];
        p[2] = q[2];
    }

    for (auto f : mesh.faces()) {
        f.data().property.cells = std::move(face_tets[f.id.idx]);
    }

    return mesh;
}

auto setup_neg_distance(tet_mesh::TetMesh& mesh, const std::vector<std::vector<gpf::FaceId>>& label_face_groups) {
    std::vector<std::vector<Triangle>> triangle_groups;
    triangle_groups.reserve(label_face_groups.size());
    for (const auto& faces : label_face_groups) {
        std::vector<Triangle> triangles;
        triangles.reserve(faces.size());
        for (const auto fid : faces) {
            auto he = mesh.face(fid).halfedge();
            auto points = mesh.face(fid).halfedges()
                | view_ts([] (auto he) {
                    auto& p = he.to().data().property.pt;
                    return Point_3(p[0], p[1], p[2]); })
                | std::ranges::to<std::vector>();
            triangles.emplace_back(std::move(points[0]), std::move(points[1]), std::move(points[2]));
        }
        triangle_groups.emplace_back(std::move(triangles));
    }

    std::vector<Tree> trees;
    trees.reserve(label_face_groups.size());
    for (const auto& triangles : triangle_groups) {
        trees.emplace_back(triangles.begin(), triangles.end());
        trees.back().accelerate_distance_queries();
    }
    for (auto v : mesh.vertices()) {
        auto& dists = v.data().property.distances;
        const auto& p = v.data().property.pt;
        dists.resize(trees.size());
        Point_3 pt(p[0], p[1], p[2]);
        for (auto [d, tree] : std::ranges::zip_view(dists, trees)) {
            d = -std::sqrt(tree.squared_distance(pt));
        }
    }
}

int main(int argc, char* argv[]) {
    CLI::App app { "multi-label-partition" };
    std::string mesh_path;
    app.add_option("-m,--mesh", mesh_path, "Path to mesh file")->required();
    CLI11_PARSE(app, argc, argv);

    std::vector<std::array<std::size_t, 3>> colors;
    auto [points, faces, label_face_groups] = read_colored_mesh(mesh_path);

    VMat V(points.size(), 3);
    FMat F(faces.size(), 3);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto& p = points[i];
        V.row(i) << p[0], p[1], p[2];
    }
    for (std::size_t i = 0; i < faces.size(); ++i) {
        const auto& f = faces[i];
        F.row(i) << f[0], f[1], f[2];
    }

    VMat TV;
    TMat TT;
    FMat TF;

    igl::copyleft::tetgen::tetrahedralize(V, F, "pq1.414Y", TV, TT, TF);
    auto tet_mesh = build_tet_mesh(TV, TT, TF);
    setup_neg_distance(tet_mesh, label_face_groups);
    write_msh("123.obj", TV, TT);
    return 0;
}
