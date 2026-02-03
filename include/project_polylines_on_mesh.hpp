#include "gpf/manifold_mesh.hpp"
#include "gpf/property.hpp"
#include <unordered_map>
#include <vector>
#include <array>
#include <span>
#include <ranges>

#include <gpf/mesh.hpp>
#include <gpf/triangulation.hpp>

#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>


namespace detail {
namespace views = std::views;
namespace ranges = std::ranges;
using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;

inline std::vector<double> compute_bary_coordinates(const std::span<const double> points) {
    auto pa = Vector2d::Map(points.data());
    auto pb = Vector2d::Map(points.data() + 2);
    auto pc = Vector2d::Map(points.data() + 4);
    const auto v0 = (pb - pa).eval();
    const auto v1 = (pc - pa).eval();
    const auto d00 = v0.squaredNorm();
    const auto d01 = v0.dot(v1);
    const auto d11 = v1.squaredNorm();
    const auto denom = d00 * d11 - d01 * d01;
    std::vector<double> bary_coords((points.size() / 2 - 3) * 3);
    std::size_t idx = 0;
    for (std::size_t i = 6; i < points.size(); i += 2) {
        const auto v2 = Vector2d::Map(points.data() + i) - pa;
        const auto d20 = v2.dot(v0);
        const auto d21 = v2.dot(v1);
        const auto b1 = (d11 * d20 - d01 * d21) / denom;
        const auto b2 = (d00 * d21 - d01 * d20) / denom;
        const auto b0 = 1.0 - b1 - b2;
        bary_coords[idx++] = b0;
        bary_coords[idx++] = b1;
        bary_coords[idx++] = b2;
    }
    return bary_coords;
}

struct CCS3d {
    std::array<double, 9> data;
    CCS3d() noexcept = default;
    CCS3d(std::array<double, 9>&& data) noexcept : data(std::move(data)) {}
    CCS3d(auto& mesh, const gpf::FaceId fid) noexcept {
        auto face = mesh.face(fid);
        auto face_vertices = face.halfedges() | views::transform([](auto he) { return he.to().id; }) | ranges::to<std::vector>();
        const auto pa = Vector3d::Map(mesh.vertex_prop(face_vertices[0]).pt.data());
        const auto pb = Vector3d::Map(mesh.vertex_prop(face_vertices[1]).pt.data());
        const auto pc = Vector3d::Map(mesh.vertex_prop(face_vertices[2]).pt.data());

        auto o = Vector3d::Map(data.data());
        auto x = Vector3d::Map(data.data() + 3);
        auto y = Vector3d::Map(data.data() + 6);
        o = pa;
        x = (pb - o).normalized();
        auto z = x.cross((pc - o)).normalized();
        y = z.cross(x).normalized();
    }

    Eigen::Vector2d uv(const double* pt) const noexcept {
        auto o = Vector3d::Map(data.data());
        auto x = Vector3d::Map(data.data() + 3);
        auto y = Vector3d::Map(data.data() + 6);
        auto v = (Vector3d::Map(pt) - o).eval();
        return {v.dot(x), v.dot(y)};
    }
};

struct FaceInfo {
    CCS3d ccs;
    std::vector<std::size_t> point_indices;
};

template<typename Mesh>
auto identify_points(
    Mesh& mesh,
    const gpf::FaceId fid,
    std::vector<std::array<double, 3>>& all_points,
    std::unordered_map<gpf::EdgeId, std::vector<std::size_t>>& edge_to_points_map,
    std::vector<std::size_t>& face_point_indices,
    std::vector<gpf::VertexId>& point_vertices,
    const double eps
) {
    auto face = mesh.face(fid);
    auto face_halfedges = face.halfedges() | views::transform([](auto he) { return he.id; }) | ranges::to<std::vector>();
    auto face_vertices = face_halfedges | views::transform([&mesh](auto hid) { return mesh.halfedge(hid).to().id; }) | ranges::to<std::vector>();
    const auto pa = Vector3d::Map(mesh.vertex_prop(face_vertices[0]).pt.data());
    const auto pb = Vector3d::Map(mesh.vertex_prop(face_vertices[1]).pt.data());
    const auto pc = Vector3d::Map(mesh.vertex_prop(face_vertices[2]).pt.data());

    std::array<double, 9> ccs_data;
    auto o = Vector3d::Map(ccs_data.data());
    auto x = Vector3d::Map(ccs_data.data() + 3);
    auto y = Vector3d::Map(ccs_data.data() + 6);
    o = pa;
    x = (pb - o).normalized();
    auto z = x.cross((pc - o)).normalized();
    y = z.cross(x).normalized();

    auto ccs = CCS3d{std::move(ccs_data)};
    std::vector<double> uvs((3 + face_point_indices.size()) * 2, 0.0);
    Vector2d::Map(uvs.data() + 2) = ccs.uv(pb.data());
    Vector2d::Map(uvs.data() + 4) = ccs.uv(pc.data());
    std::size_t idx = 6;
    for (const auto pid : face_point_indices) {
        Vector2d::Map(uvs.data() + idx) = ccs.uv(all_points[pid].data());
        idx += 2;
    }
    auto bary_coords = compute_bary_coordinates(uvs);

    idx = 0;
    for (std::size_t i = 0; i < face_point_indices.size(); i++) {
        const auto pid = face_point_indices[i];
        Vector3d bary_coord = Vector3d::Map(bary_coords.data() + i * 3).array().max(0.0).min(1.0);
        Eigen::Index _, min_idx, max_idx;
        bary_coord.maxCoeff(&max_idx, &_);
        bary_coord.minCoeff(&min_idx, &_);
        if (std::abs(bary_coord[max_idx] - 1.0) < eps) {
            const auto vid = face_vertices[max_idx];
            point_vertices[pid] = vid;
            all_points[pid] = mesh.vertex_prop(vid).pt;
        } else if (std::abs(bary_coord[min_idx]) < eps) {
            const auto hid = face_halfedges[(min_idx + 2) % 3];
            const auto eid = mesh.he_edge(hid);
            const auto [v1, v2] = mesh.he_vertices(hid);
            auto j = (min_idx + 1) % 3;
            auto k = (min_idx + 2) % 3;
            assert(v1 == face_vertices[j]);
            assert(v2 == face_vertices[k]);

            auto sum = bary_coord[j] + bary_coord[k];
            auto t = bary_coord[j] / sum;
            Vector3d::Map(all_points[pid].data()) =
                t * Vector3d::Map(mesh.vertex_prop(v1).pt.data()) + (1.0 - t) * Vector3d::Map(mesh.vertex_prop(v2).pt.data());
            edge_to_points_map[eid].emplace_back(pid);
        } else {
            if (idx != i) {
                face_point_indices[idx] = pid;
            }
            idx += 1;
        }
    }
    face_point_indices.resize(idx);
    return ccs;
}


template<typename Mesh>
void split_edge_by_points(
    Mesh& mesh,
    const gpf::EdgeId eid,
    const std::vector<std::array<double, 3>>& all_points,
    const std::vector<std::size_t>& edge_point_indices,
    std::vector<gpf::VertexId>& point_vertices,
    const double eps
) {
    if (edge_point_indices.size() == 1) {
        const auto new_vid = mesh.split_edge(eid);
        const auto pid = edge_point_indices[0];
        mesh.vertex_prop(new_vid).pt = all_points[pid];
        point_vertices[pid] = new_vid;
    } else {
        auto curr_hid = mesh.edge(eid).halfedge().id;
        const auto [va, vb] = mesh.he_vertices(curr_hid);
        const auto pa = mesh.vertex_prop(va).pt;
        const auto pb = mesh.vertex_prop(vb).pt;
        const auto pa_ref = Vector3d::Map(pa.data());
        const auto pb_ref = Vector3d::Map(pb.data());
        auto edge_len = (pb_ref - pa_ref).norm();
        std::vector<std::size_t> indices(edge_point_indices.size());
        std::iota(indices.begin(), indices.end(), 0);
        const auto distances = edge_point_indices |
            std::views::transform([pa_ref, edge_len, &all_points](const auto pid) {
                return (Vector3d::Map(all_points[pid].data()) - pa_ref).norm() / edge_len;
            }) | std::ranges::to<std::vector>();

        std::sort(indices.begin(), indices.end(), [&distances](auto i, auto j) { return distances[i] < distances[j]; });
        std::size_t j = 0;
        gpf::EdgeId curr_eid;
        for (std::size_t i = 0; i < indices.size(); i++) {
            const auto pid = edge_point_indices[indices[i]];
            if (i == 0 || (distances[indices[i]] - distances[indices[j]]) > eps) {
                const auto new_vid = mesh.split_edge(curr_eid);
                auto new_v = mesh.vertex(new_vid);
                new_v.prop().pt = all_points[pid];
                point_vertices[pid] = new_vid;
                curr_eid = new_v.halfedge().edge().id;
                j = i;
            } else {
                point_vertices[pid] = point_vertices[edge_point_indices[indices[j]]];
            }
        }
    }
}
template<typename Mesh>
void triangulate_on_face(
    Mesh& mesh,
    const gpf::FaceId fid,
    const std::vector<std::array<double, 3>>& all_points,
    const CCS3d& ccs,
    const std::vector<std::size_t>& point_indices,
    std::vector<gpf::VertexId>& point_vertices
) {
    auto face_vertices = mesh.face(fid).halfedges() |
        views::transform([](const auto& he) { return he.to().id; }) |
        ranges::to<std::vector>();
    const auto n_old_face_vertices = face_vertices.size();
    const auto n_old_vertices = mesh.n_vertices_capacity();
    mesh.new_vertices(point_indices.size());
    face_vertices.append_range(ranges::iota_view{n_old_vertices, mesh.n_vertices_capacity()} | views::transform([](const auto idx) { return gpf::VertexId{idx}; }));
    for (auto [pid, vid] : views::zip(point_indices, ranges::drop_view{face_vertices, static_cast<std::ptrdiff_t>(n_old_face_vertices)})) {
        mesh.vertex_prop(vid).pt = all_points[pid];
        point_vertices[pid] = vid;
    }

    std::vector<double> points(face_vertices.size() * 2);
    std::size_t idx = 0;
    for (auto vid : face_vertices) {
        const auto& pt = mesh.vertex_prop(vid).pt;
        Vector2d::Map(points.data() + idx) = ccs.uv(pt.data());
        idx += 2;
    }

    std::vector<std::size_t> contour;
    contour.reserve(n_old_face_vertices * 2);
    for (std::size_t i = 0; i < n_old_face_vertices; i++) {
        contour.push_back(i);
        contour.push_back((i + 1) % n_old_face_vertices);
    }

    const auto triangle_indices = gpf::triangulate_polygon(points, contour, true);
    auto triangles = triangle_indices | views::transform([&face_vertices] (const auto idx) { return face_vertices[idx];}) | ranges::to<std::vector>();
    mesh.split_face_into_triangles(fid, triangles);
}

template<typename VP, typename HP, typename EP, typename FP>
void project_points_on_mesh(
    std::vector<std::array<double, 3>>& points,
    gpf::ManifoldMesh<VP, HP, EP, FP>& mesh
) {
    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Point_3 = Kernel::Point_3;
    using Triangle_3 = Kernel::Triangle_3;
    using TreeIterator = std::vector<Triangle_3>::const_iterator;
    using TreePrimitive = CGAL::AABB_triangle_primitive_3<Kernel, TreeIterator>;
    using TreeTraits = CGAL::AABB_traits_3<Kernel, TreePrimitive>;
    using Tree = CGAL::AABB_tree<TreeTraits>;

    std::vector<Triangle_3> triangles;
    std::vector<gpf::FaceId> face_ids;
    triangles.reserve(mesh.n_faces());
    face_ids.reserve(mesh.n_faces());

    std::array<Point_3, 3> tri;
    for (auto face : mesh.faces()) {
        auto he = face.halfedge();
        for (std::size_t i = 0; i < 3ul; ++i) {
            const auto& p = he.to().prop().pt;
            tri[i] = Point_3(p[0], p[1], p[2]);
            he = he.next();
        }
        triangles.push_back(Triangle_3(tri[0], tri[1], tri[2]));
        face_ids.emplace_back(face.id);
    }

    Tree tree(triangles.begin(), triangles.end());
    tree.accelerate_distance_queries();
    std::unordered_map<gpf::FaceId, FaceInfo> face_info_map;
    for (std::size_t pid = 0; pid < points.size(); pid++) {
        const auto& pt = points[pid];
        Point_3 p(pt[0], pt[1], pt[2]);

        auto closest_ret = tree.closest_point_and_primitive(p);
        auto fid = face_ids[std::distance(triangles.cbegin(), closest_ret.second)];
        face_info_map[fid].point_indices.emplace_back(pid);
    }

    constexpr double EPS = 1e-3;
    std::vector<gpf::VertexId> point_vertices(points.size(), gpf::VertexId{});
    std::unordered_map<gpf::EdgeId, std::vector<std::size_t>> edge_to_points_map;
    for (auto& [fid, info] : face_info_map) {
        info.ccs = identify_points(mesh, fid, points, edge_to_points_map, info.point_indices, point_vertices, EPS);
    }

    for (const auto eid : edge_to_points_map | views::keys) {
        for (const auto he : mesh.edge(eid).halfedges()) {
            const auto fid = he.face().id;
            if (face_info_map.find(fid) == face_info_map.end()) {
                face_info_map.emplace(fid, FaceInfo{.ccs = CCS3d(mesh, fid)});
            }
        }
    }

    for (const auto& [eid, point_indices] : edge_to_points_map) {
        split_edge_by_points(mesh, eid, points, point_indices, point_vertices, EPS);
    }

    for (const auto& [fid, info] : face_info_map) {
        triangulate_on_face(mesh, fid, points, info.ccs, info.point_indices, point_vertices);
    }
}

struct VertexProp {
    double angle_sum;
};

struct EdgeProp {
    double len;
};

struct HalfedgeProp {
    double angle;
};

using AuxiliaryMesh = gpf::ManifoldMesh<VertexProp, HalfedgeProp, EdgeProp, gpf::Empty>;

}

template<typename VP, typename HP, typename EP, typename FP>
auto project_polylines_on_mesh(
    std::vector<std::array<double, 3>>& points,
    const std::vector<std::vector<std::size_t>>& polylines,
    gpf::ManifoldMesh<VP, HP, EP, FP>& mesh
) {
    detail::project_points_on_mesh(points, mesh);
    detail::AuxiliaryMesh aux_mesh;
    aux_mesh.copy_from(mesh);
    const auto a = 2;
}
