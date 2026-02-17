#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <predicates/predicates.hpp>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#include <array>
#include <span>
#include <ranges>

#include <Eigen/Dense>
#include <Eigen/Geometry>

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

// Complex division: treat 2D vectors as complex numbers (x + yi)
inline Vector2d complex_div(const Vector2d& a, const Vector2d& b) {
    double denom = b.squaredNorm();
    return Vector2d{
        (a.x() * b.x() + a.y() * b.y()) / denom,
        (a.y() * b.x() - a.x() * b.y()) / denom
    };
}
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


template<typename Mesh, typename PointAccessor>
void split_edge_by_points(
    Mesh& mesh,
    const gpf::EdgeId eid,
    PointAccessor&& get_point,
    const std::vector<std::size_t>& edge_point_indices,
    std::vector<gpf::VertexId>& point_vertices,
    const double eps,
    std::unordered_map<gpf::EdgeId, gpf::EdgeId>* edge_parent_map = nullptr
) {
    // Find the root parent for this edge
    auto get_root_parent = [&edge_parent_map](gpf::EdgeId e) {
        gpf::EdgeId root = e;
        if (edge_parent_map) {
            auto it = edge_parent_map->find(e);
            if (it != edge_parent_map->end()) {
                root = it->second;
            }
        }
        return root;
    };

    if (edge_point_indices.size() == 1) {
        const auto root_parent = get_root_parent(eid);
        const auto new_vid = mesh.split_edge(eid);
        const auto pid = edge_point_indices[0];
        mesh.vertex_prop(new_vid).pt = get_point(pid);
        point_vertices[pid] = new_vid;
        if (edge_parent_map) {
            // eid is reused for one sub-edge, new edge is the other
            (*edge_parent_map)[eid] = root_parent;
            auto new_eid = mesh.vertex(new_vid).halfedge().edge().id;
            (*edge_parent_map)[new_eid] = root_parent;
        }
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
            std::views::transform([pa_ref, edge_len, &get_point](const auto pid) {
                auto pt = get_point(pid);
                return (Vector3d::Map(pt.data()) - pa_ref).norm() / edge_len;
            }) | std::ranges::to<std::vector>();

        std::sort(indices.begin(), indices.end(), [&distances](auto i, auto j) { return distances[i] < distances[j]; });
        const auto root_parent = get_root_parent(eid);
        std::size_t j = 0;
        gpf::EdgeId curr_eid = eid;
        for (std::size_t i = 0; i < indices.size(); i++) {
            const auto pid = edge_point_indices[indices[i]];
            if (i == 0 || (distances[indices[i]] - distances[indices[j]]) > eps) {
                const auto new_vid = mesh.split_edge(curr_eid);
                auto new_v = mesh.vertex(new_vid);
                new_v.prop().pt = get_point(pid);
                point_vertices[pid] = new_vid;
                if (edge_parent_map) {
                    auto new_eid = new_v.halfedge().edge().id;
                    (*edge_parent_map)[new_eid] = root_parent;
                }
                curr_eid = new_v.halfedge().edge().id;
                j = i;
            } else {
                point_vertices[pid] = point_vertices[edge_point_indices[indices[j]]];
            }
        }
        if (edge_parent_map) {
            (*edge_parent_map)[eid] = root_parent;
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
    const std::vector<gpf::VertexId>& cross_edge_vertices,
    std::vector<gpf::VertexId>& point_vertices,
    std::unordered_map<gpf::FaceId, gpf::FaceId>* face_parent_map = nullptr
) {
    auto face_vertices = mesh.face(fid).halfedges() |
        views::transform([](const auto& he) { return he.to().id; }) |
        ranges::to<std::vector>();

    std::unordered_map<gpf::VertexId, std::size_t> vertex_indices;
    for (auto vid : cross_edge_vertices) {
        vertex_indices.emplace(vid, gpf::kInvalidIndex);
    }
    for (std::size_t i = 0; i < face_vertices.size(); i++) {
        const auto vid = face_vertices[i];
        auto it = vertex_indices.find(vid);
        if (it != vertex_indices.end()) {
            it->second = i;
        }
    }
    const auto n_old_face_vertices = face_vertices.size();
    const auto n_old_vertices = mesh.n_vertices_capacity();
    mesh.new_vertices(point_indices.size());
    face_vertices.append_range(ranges::iota_view{n_old_vertices, mesh.n_vertices_capacity()} | views::transform([](const auto idx) { return gpf::VertexId{idx}; }));
    if (face_vertices.size() == 3) {
        return;
    }
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

    std::vector<std::size_t> segments;
    segments.reserve(n_old_face_vertices * 2 + cross_edge_vertices.size());
    for (std::size_t i = 0; i < n_old_face_vertices; i++) {
        segments.push_back(i);
        segments.push_back((i + 1) % n_old_face_vertices);
    }

    segments.append_range(cross_edge_vertices | views::transform([&vertex_indices] (const auto vid) { return vertex_indices[vid]; }));

    const auto triangle_indices = gpf::triangulate_polygon(points, segments, n_old_face_vertices, true);
    auto triangles = triangle_indices | views::transform([&face_vertices] (const auto idx) { return face_vertices[idx];}) | ranges::to<std::vector>();
    assert(!triangles.empty());

    const auto n_faces_before = mesh.n_faces_capacity();
    mesh.split_face_into_triangles(fid, triangles);

    if (face_parent_map) {
        // Find the root parent (in case fid was already a sub-triangle)
        gpf::FaceId root_parent = fid;
        auto it = face_parent_map->find(fid);
        if (it != face_parent_map->end()) {
            root_parent = it->second;
        }

        // The original face fid is reused for one of the sub-triangles
        (*face_parent_map)[fid] = root_parent;
        // New faces are created starting from n_faces_before
        for (std::size_t i = n_faces_before; i < mesh.n_faces_capacity(); ++i) {
            (*face_parent_map)[gpf::FaceId{i}] = root_parent;
        }
    }
}

template<typename VP, typename HP, typename EP, typename FP>
auto project_points_on_mesh(
    std::vector<std::array<double, 3>>& points,
    gpf::ManifoldMesh<VP, HP, EP, FP>& mesh,
    const double eps,
    std::unordered_map<gpf::FaceId, gpf::FaceId>* face_parent_map = nullptr,
    std::unordered_map<gpf::EdgeId, gpf::EdgeId>* edge_parent_map = nullptr
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

    std::vector<gpf::VertexId> point_vertices(points.size(), gpf::VertexId{});
    std::unordered_map<gpf::EdgeId, std::vector<std::size_t>> edge_to_points_map;
    for (auto& [fid, info] : face_info_map) {
        info.ccs = identify_points(mesh, fid, points, edge_to_points_map, info.point_indices, point_vertices, eps);
    }

    // add ccs for all triangles sharing this edge to prepare triangulation
    for (const auto eid : edge_to_points_map | views::keys) {
        for (const auto he : mesh.edge(eid).halfedges()) {
            const auto fid = he.face().id;
            if (face_info_map.find(fid) == face_info_map.end()) {
                face_info_map.emplace(fid, FaceInfo{.ccs = CCS3d(mesh, fid)});
            }
        }
    }

    for (const auto& [eid, point_indices] : edge_to_points_map) {
        split_edge_by_points(mesh, eid, [&points](std::size_t pid) { return points[pid]; }, point_indices, point_vertices, eps, edge_parent_map);
    }

    for (const auto& [fid, info] : face_info_map) {
        triangulate_on_face(mesh, fid, points, info.ccs, info.point_indices, {}, point_vertices, face_parent_map);
    }
    return point_vertices;
}

struct VertexProp {
    double angle_sum = 0.0;
};

struct EdgeProp {
    double len = 0.0;
    bool locked = false;
    bool is_origin = true;
};

struct HalfedgeProp {
    std::array<double, 2> vector;
    double angle = 0.0;
    double signpost_angle = 0.0;
    gpf::HalfedgeId path_prev{};
    gpf::HalfedgeId path_next{};

    void unconnect() {
        path_prev = {};
        path_next = {};
    }
};

using AuxiliaryMesh = gpf::ManifoldMesh<VertexProp, HalfedgeProp, EdgeProp, gpf::Empty>;

template<typename Mesh, typename IsLocked, typename GetEdgeLength>
inline std::vector<gpf::HalfedgeId> shortest_patch_by_dijksta(
    const Mesh& mesh,
    const gpf::VertexId start_vid,
    const gpf::VertexId end_vid,
    IsLocked&& is_locked,
    GetEdgeLength&& get_edge_length
) {
    for (const auto he :  mesh.vertex(start_vid).outgoing_halfedges()) {
        if (he.to().id == end_vid && !is_locked(he.edge())) {
            return {he.id};
        }
    }

    using WeightedHalfedge = std::pair<double, gpf::HalfedgeId>;
    std::unordered_map<gpf::VertexId, gpf::HalfedgeId> incomindg_halfedges_map;
    std::priority_queue<WeightedHalfedge, std::vector<WeightedHalfedge>, std::greater<WeightedHalfedge>> pq;
    auto vertex_used = [start_vid, &incomindg_halfedges_map](const gpf::VertexId vid) {
        return vid == start_vid || incomindg_halfedges_map.find(vid) != incomindg_halfedges_map.end();
    };

    auto enqueue_vertex_neighbors = [&mesh, &is_locked, &get_edge_length, &vertex_used, &pq](const gpf::VertexId vid, double dist) {
        for (const auto he : mesh.vertex(vid).outgoing_halfedges()) {
            if (is_locked(he.edge())) {
                continue;
            }
            auto vb = he.to().id;
            if (!vertex_used(vb)) {
                auto len = get_edge_length(he.edge());
                pq.emplace(dist + len, he.id);
            }
        }
    };

    enqueue_vertex_neighbors(start_vid, 0.0);
    while(!pq.empty()) {
        auto [curr_dist, hid] = pq.top();
        pq.pop();
        auto vid = mesh.he_to(hid);
        if (vertex_used(vid)) {
            continue;
        }
        if (vid == end_vid) {
            std::vector<gpf::HalfedgeId> path;
            do {
                path.emplace_back(hid);
                auto vid = mesh.he_from(hid);
                if (vid == start_vid) {
                    break;
                }
                assert(incomindg_halfedges_map.find(vid) != incomindg_halfedges_map.end());
                hid = incomindg_halfedges_map[vid];
            } while(true);
            ranges::reverse(path);
            return path;
        }
        incomindg_halfedges_map.emplace(vid, hid);
        enqueue_vertex_neighbors(vid, curr_dist);
    }
    return {};
}

inline auto triangle_oppo_point(
    const double lab,
    const double lbc,
    const double lca,
    bool reversed
) {
    const auto h = gpf::triangle_area(lab, lbc, lca) / lab * 2.0;
    if (reversed) {
        auto w = (lbc * lbc + lab * lab - lca * lca) / lab * 0.5;
        return Eigen::Vector2d{w, -h};
    } else {
        const auto w = (lca * lca + lab * lab - lbc * lbc) / lab * 0.5;
        return Eigen::Vector2d{w, h};
    }
}

struct FlipGeodesic {
    enum class TurnDirection {
        Left,
        Right
    };
    struct Wedge {
        double angle;
        std::array<gpf::VertexId, 3> vertices;
        gpf::HalfedgeId hid;
        TurnDirection dir;

        auto operator<=>(const Wedge& other) const noexcept {
            return angle <=> other.angle;
        }
        bool operator==(const Wedge& other) const noexcept = default;
    };

    std::vector<gpf::HalfedgeId> perform(std::vector<gpf::HalfedgeId>&& raw_path);

    void init_wedge_queue(const std::vector<gpf::HalfedgeId>& raw_path);
    void add_wedge(const gpf::HalfedgeId ha, const gpf::HalfedgeId hb);
    void shorten_locally();
    bool flip(const gpf::HalfedgeId hid) const;
    void replace_path(std::vector<gpf::HalfedgeId>&& new_path, const gpf::HalfedgeId path_prev_prev_hid, const gpf::HalfedgeId path_next_next_hid);

    AuxiliaryMesh* mesh;
    std::priority_queue<Wedge, std::vector<Wedge>, std::greater<Wedge>> pq{};
    gpf::VertexId path_start_vid{};
    gpf::VertexId path_end_vid{};
    gpf::HalfedgeId path_start_hid{};
};

inline std::vector<gpf::HalfedgeId> FlipGeodesic::perform(std::vector<gpf::HalfedgeId>&& raw_path) {
    if (raw_path.size() == 1) {
        assert(!mesh->halfedge(raw_path.back()).edge().prop().locked);
        assert(mesh->halfedge(raw_path.back()).edge().prop().is_origin);
        return raw_path;
    }
    init_wedge_queue(raw_path);
    while(!pq.empty()) {
        shorten_locally();
    }
    std::vector<gpf::HalfedgeId> result;
    result.reserve(raw_path.size());
    auto curr_hid = path_start_hid;
    while (true) {
        result.emplace_back(curr_hid);
        mesh->halfedge(curr_hid).edge().prop().locked = true;
        if (mesh->he_to(curr_hid) == path_end_vid) {
            break;
        }

        curr_hid = mesh->halfedge_prop(curr_hid).path_next;
        assert(curr_hid.valid());
    }
    return result;
}


inline void FlipGeodesic::init_wedge_queue(const std::vector<gpf::HalfedgeId>& raw_path) {
    path_start_vid = mesh->he_from(raw_path.front());
    path_start_hid = raw_path.front();
    path_end_vid = mesh->he_to(raw_path.back());
    for (std::size_t i = 0; i + 1 < raw_path.size(); i++) {
        auto h1 = raw_path[i];
        auto h2 = raw_path[i + 1];
        mesh->halfedge_prop(h1).path_next = h2;
        mesh->halfedge_prop(h2).path_prev = h1;
        add_wedge(h1, h2);
    }
}

inline void FlipGeodesic::add_wedge(const gpf::HalfedgeId h1, const gpf::HalfedgeId h2) {
    auto ha = mesh->halfedge(h1);
    auto hb = mesh->halfedge(h2);
    auto vb = ha.to();
    auto angle_sum = vb.prop().angle_sum;
    auto angle_in = ha.twin().prop().signpost_angle;
    auto angle_out = hb.prop().signpost_angle;
    auto is_boundary = mesh->v_is_boundary(vb.id);
    double right_angle = std::numeric_limits<double>::infinity();
    double left_angle = std::numeric_limits<double>::infinity();
    if (angle_in < angle_out) {
        right_angle = angle_out - angle_in;
    } else if (!is_boundary) {
        right_angle = angle_sum - angle_in + angle_out;
    }

    if (angle_out < angle_in) {
        left_angle = angle_in - angle_out;
    } else if (!is_boundary) {
        left_angle = angle_sum - angle_out + angle_in;
    }

    auto va = ha.from().id;
    auto vc = hb.to().id;
    constexpr double EPS_ANGLE = 1e-5;
    if (left_angle < std::numbers::pi - EPS_ANGLE) {
        pq.push(Wedge{.angle = left_angle, .vertices{va, vb.id, vc}, .hid = h1, .dir = TurnDirection::Left});
    }
    if (right_angle < std::numbers::pi - EPS_ANGLE) {
        pq.push(Wedge{.angle = right_angle, .vertices{va, vb.id, vc}, .hid = h1, .dir = TurnDirection::Right});
    }
}

inline void FlipGeodesic::shorten_locally() {
    auto [angle, vertices, path_prev_hid, dir] = pq.top();
    auto path_next_hid = mesh->halfedge_prop(path_prev_hid).path_next;
    pq.pop();
    if (
        !path_next_hid.valid() ||
        mesh->he_from(path_prev_hid) != vertices[0] ||
        mesh->he_to(path_prev_hid) != vertices[1] ||
        mesh->he_to(path_next_hid) != vertices[2]
    ) {
        return;
    }

    const auto path_prev_prev_hid = mesh->halfedge_prop(path_prev_hid).path_prev;
    const auto path_next_next_hid = mesh->halfedge_prop(path_next_hid).path_next;

    auto [prev_hid, next_hid] = dir == TurnDirection::Left ? std::pair{path_prev_hid, path_next_hid} : std::pair{mesh->he_twin(path_next_hid), mesh->he_twin(path_prev_hid)};
    auto curr_he = mesh->halfedge(prev_hid).next();
    while(curr_he.id != next_hid) {
        if (curr_he.twin().id == prev_hid) {
            curr_he = curr_he.twin().next();
            continue;
        }
        if (flip(curr_he.id)) {
            curr_he = curr_he.next().twin();
        } else {
            curr_he = curr_he.twin().next();
        }
    }

    std::vector<gpf::HalfedgeId> new_path;
    curr_he = mesh->halfedge(prev_hid).next();
    while (true) {
        new_path.emplace_back(curr_he.next().twin().id);
        if (curr_he.id == next_hid) {
            break;
        }
        curr_he = curr_he.twin().next();
    }
    if (dir == TurnDirection::Right) {
        ranges::reverse(new_path);
        for (auto& hid : new_path) {
            hid = mesh->he_twin(hid);
        }
    }
    mesh->halfedge_prop(path_prev_hid).unconnect();
    mesh->halfedge_prop(path_next_hid).unconnect();
    replace_path(std::move(new_path), path_prev_prev_hid, path_next_next_hid);
}

inline bool FlipGeodesic::flip(const gpf::HalfedgeId hid) const {
    if (mesh->halfedge(hid).edge().prop().locked) {
        return false;
    }

    auto hac = hid;
    auto hca = mesh->he_twin(hac);
    auto hcd = mesh->he_next(hac);
    auto hda = mesh->he_next(hcd);
    auto hab = mesh->he_next(hca);
    auto hbc = mesh->he_next(hab);

    auto lab = mesh->halfedge(hab).edge().prop().len;
    auto lbc = mesh->halfedge(hbc).edge().prop().len;
    auto lcd = mesh->halfedge(hcd).edge().prop().len;
    auto lda = mesh->halfedge(hda).edge().prop().len;
    auto lca = mesh->halfedge(hca).edge().prop().len;

    Eigen::Vector2d pa {lca, 0.0};
    auto pb = triangle_oppo_point(lca, lab, lbc, false);
    auto pd = triangle_oppo_point(lca, lcd, lda, true);

    auto left_area = pd.cross(pb); // pc = {0.0, 0.0}
    auto right_area = (pb - pa).cross(pd - pa);
    constexpr double TRIANGLE_TEST_EPS = 1e-6;
    auto area_sum = left_area + right_area;
    if(left_area / area_sum < TRIANGLE_TEST_EPS || right_area / area_sum < TRIANGLE_TEST_EPS) {
        return false;
    }

    mesh->flip(hid);
    auto he_ab = mesh->halfedge(hab);
    auto he_bc = mesh->halfedge(hbc);
    auto he_cd = mesh->halfedge(hcd);
    auto he_da = mesh->halfedge(hda);
    auto he_bd = mesh->halfedge(hid);
    auto he_db = he_bd.twin();
    assert(he_bd.next().id == he_da.id);

    he_bd.edge().prop().len = (pd - pb).norm();
    gpf::update_corner_angles_on_face(he_bd.face());
    gpf::update_corner_angles_on_face(he_db.face());

    // edge bd counld't be boundary
    he_bd.prop().signpost_angle = std::fmod(he_bc.prop().signpost_angle + he_bc.prop().angle, he_ab.to().prop().angle_sum);
    gpf::update_halfedge_vector(he_bd);
    he_db.prop().signpost_angle = std::fmod(he_da.prop().signpost_angle + he_da.prop().angle, he_cd.to().prop().angle_sum);
    gpf::update_halfedge_vector(he_db);

    // auto _a1 = std::fmod(he_ab.twin().prop().signpost_angle - he_bd.prop().angle, he_bd.from().prop().angle_sum);
    // auto _a2 = he_bd.prop().signpost_angle - _a1;
    assert(std::abs(he_db.prop().signpost_angle - std::fmod(he_cd.twin().prop().signpost_angle - he_db.prop().angle + he_db.from().prop().angle_sum, he_db.from().prop().angle_sum)) < 1e-8);
    assert(std::abs(he_bd.prop().signpost_angle - std::fmod(he_ab.twin().prop().signpost_angle - he_bd.prop().angle + he_bd.from().prop().angle_sum, he_bd.from().prop().angle_sum)) < 1e-8);
    assert(std::abs(ranges::fold_left( he_bd.from().outgoing_halfedges() | views::transform([](auto he) { return he.prop().angle; }), 0.0, std::plus{}) - he_bd.from().prop().angle_sum) < 1e-8);
    assert(std::abs(ranges::fold_left( he_db.from().outgoing_halfedges() | views::transform([](auto he) { return he.prop().angle; }), 0.0, std::plus{}) - he_db.from().prop().angle_sum) < 1e-8);
    return true;
}

inline void FlipGeodesic::replace_path(std::vector<gpf::HalfedgeId>&& new_path, const gpf::HalfedgeId path_prev_prev_hid, const gpf::HalfedgeId path_next_next_hid) {
    auto prev_he = mesh->halfedge(path_prev_prev_hid);
    auto curr_he = mesh->halfedge(new_path.front());
    curr_he.prop().path_prev = prev_he.id;
    if (prev_he.id.valid()) {
        prev_he.prop().path_next = curr_he.id;
        add_wedge(prev_he.id, curr_he.id);
    } else if (mesh->he_from(curr_he.id) == path_start_vid) {
        path_start_hid = curr_he.id;
    }

    for (std::size_t i = 1; i < new_path.size(); ++i) {
        prev_he = curr_he;
        curr_he = mesh->halfedge(new_path[i]);
        curr_he.prop().path_prev = prev_he.id;
        prev_he.prop().path_next = curr_he.id;
        add_wedge(prev_he.id, curr_he.id);
    }
    curr_he.prop().path_next = path_next_next_hid;
    if (path_next_next_hid.valid()) {
        mesh->halfedge_prop(path_next_next_hid).path_prev = curr_he.id;
        add_wedge(curr_he.id, path_next_next_hid);
    }
}

template<std::size_t N, typename Mesh>
struct TracePolyline {
    struct EdgePoint {
        gpf::EdgeId eid;
        double t;
        std::array<double, N> pt;
    };

    using Anchor = std::variant<gpf::VertexId, std::size_t>;

    void trace_from_vertex(gpf::HalfedgeId start_hid);
    void trace_from_edge(gpf::HalfedgeId hab, const double* dir, const gpf::VertexId end_vid);
    void add_intersection_point(double left_ori, double right_ori, gpf::HalfedgeId hab);
    std::span<const double> origin_signpost_angles;
    std::span<const double> origin_edge_lengths;
    std::vector<EdgePoint>& edge_points;
    Mesh* mesh;
    AuxiliaryMesh* aux_mesh;
    std::vector<Anchor> path;
    std::vector<gpf::FaceId> path_on_face_vec;
};

template<std::size_t N, typename Mesh>
void TracePolyline<N, Mesh>::trace_from_vertex(gpf::HalfedgeId start_hid) {
    auto start_vid = aux_mesh->he_from(start_hid);
    assert(std::get<gpf::VertexId>(path.back()) == start_vid);
    auto end_vid = aux_mesh->he_to(start_hid);
    for (auto he : mesh->vertex(start_vid).outgoing_halfedges()) {
        if (he.to().id == end_vid) {
            path.emplace_back(end_vid);
            path_on_face_vec.emplace_back();
            return;
        }
    }

    if constexpr (N == 2) {
        auto vc = mesh->vertex(start_vid);
        auto prev_he = vc.halfedge().prev().twin();
        const auto first_hid = prev_he.id;
        const auto& pc = vc.prop().pt;
        auto pa = prev_he.to().prop().pt;
        const auto& pd = mesh->vertex(end_vid).prop().pt;
        auto right_ori = predicates::orient2d(pc.data(), pa.data(), pd.data());
        while(true) {
            auto curr_he = prev_he.prev().twin();

            const auto& pb = curr_he.to().prop().pt;
            auto left_ori = predicates::orient2d(pc.data(), pd.data(), pb.data());
            if (right_ori >= 0.0 && left_ori > 0.0) {
                add_intersection_point(left_ori, right_ori, prev_he.next().id);
                const Vector2d dir = Vector2d::Map(pd.data()) - Vector2d::Map(pc.data());
                trace_from_edge(prev_he.next().twin().id, dir.data(), end_vid);
                return;
            } else {
                prev_he = curr_he;
                if (prev_he.id == first_hid) {
                    // never arrive here
                    break;
                }
                right_ori = -left_ori;
                pa = pb;
            }
        }
    } else {
        auto get_orientations = [this](gpf::HalfedgeId hca, gpf::HalfedgeId hcb, double angle) {
            const auto hab = mesh->he_next(hca);
            auto lab = origin_edge_lengths[mesh->he_edge(hab).idx];
            auto lbc = origin_edge_lengths[mesh->he_edge(hcb).idx];
            auto lca = origin_edge_lengths[mesh->he_edge(hca).idx];

            Vector2d pa{lab, 0.0};
            constexpr std::array<double, 2> pb{{0.0, 0.0}};
            Vector2d pc = triangle_oppo_point(lab, lbc, lca, true);
            std::array<double, 2> dir_arr;
            auto dir = Vector2d::Map(dir_arr.data());
            dir = Eigen::Rotation2D<double>(angle) * (pa - pc).normalized();
            auto pd = (pc + dir).eval();
            auto left_ori = predicates::orient2d(pc.data(), pd.data(), pb.data());
            auto right_ori = predicates::orient2d(pc.data(), pa.data(), pd.data());
            assert(left_ori > 0.0 && right_ori > 0.0);
            return std::make_tuple(left_ori, right_ori, std::move(dir_arr));
        };
        const auto signpost_angle = aux_mesh->halfedge(start_hid).prop().signpost_angle;
        auto prev_he = mesh->vertex(start_vid).halfedge().prev().twin();
        const auto first_hid = prev_he.id;
        auto angle_sum = aux_mesh->vertex(start_vid).prop().angle_sum;
        while (true) {
            auto curr_he = prev_he.prev().twin();
            auto in_angle = origin_signpost_angles[prev_he.id.idx];
            auto out_angle = origin_signpost_angles[curr_he.id.idx];
            if (out_angle < in_angle) {
                out_angle += angle_sum;
            }
            auto curr_angle = signpost_angle;
            if (curr_angle < in_angle) {
                curr_angle += angle_sum;
            }
            if (curr_angle >= in_angle && curr_angle < out_angle) {
                auto [left_ori, right_ori, dir] = get_orientations(prev_he.id, curr_he.id, curr_angle - in_angle);
                add_intersection_point(left_ori, right_ori, prev_he.next().id);
                trace_from_edge(prev_he.next().twin().id, dir.data(), end_vid);
                return;
            } else {
                prev_he = curr_he;
                if (prev_he.id == first_hid) {
                    // never arrive here
                    break;
                }
            }
        }
    }
}

template<std::size_t N, typename Mesh>
void TracePolyline<N, Mesh>::trace_from_edge(gpf::HalfedgeId hab, const double* dir_data, const gpf::VertexId end_vid) {
    using Vec = Eigen::Matrix<double, N, 1>;
    auto he_ab = mesh->halfedge(hab);
    auto he_bc = he_ab.next();
    if (he_bc.to().id == end_vid) {
        path.emplace_back(end_vid);
        path_on_face_vec.emplace_back(he_bc.face().id);
        return;
    }

    auto trace_next = [this, dir_data, end_vid, &he_bc](const auto& mid_pt, const auto& pa, const auto& pb, const auto& pc, const auto& pd) {
        auto vc_ori = predicates::orient2d(mid_pt.data(), pd.data(), pc.data());
        if (vc_ori > 0.0) {
            auto right_ori = predicates::orient2d(mid_pt.data(), pb.data(), pd.data());
            assert(right_ori > 0.0);
            add_intersection_point(vc_ori, right_ori, he_bc.id);
            if constexpr (N == 2) {
                trace_from_edge(he_bc.twin().id, dir_data, end_vid);
            } else {
                auto dir = Vector2d::Map(dir_data);
                Vector2d new_dir = complex_div(dir, pb - pc).normalized();
                trace_from_edge(he_bc.twin().id, new_dir.data(), end_vid);
            }
        } else {
            auto right_ori = -vc_ori;
            auto left_ori = predicates::orient2d(mid_pt.data(), pd.data(), pa.data());
            assert(left_ori > 0.0);
            auto he_ca = he_bc.next();
            add_intersection_point(left_ori, right_ori, he_bc.next().id);
            if constexpr (N == 2) {
                trace_from_edge(he_ca.twin().id, dir_data, end_vid);
            } else {
                auto dir = Vector2d::Map(dir_data);
                Vector2d new_dir = complex_div(dir, pc - pa).normalized();
                trace_from_edge(he_ca.twin().id, new_dir.data(), end_vid);
            }
        }
    };

    if constexpr (N == 2) {
        auto mid_pt = Vec::Map(edge_points.back().pt.data());
        auto pa = Vec::Map(he_ab.from().prop().pt.data());
        auto pb = Vec::Map(he_ab.to().prop().pt.data());
        auto pc = Vec::Map(he_bc.to().prop().pt.data());
        auto pd = Vec::Map(mesh->vertex(end_vid).prop().pt.data());
        trace_next(mid_pt, pa, pb, pc, pd);
    } else {
        auto he_bc = he_ab.next();
        auto he_ca = he_bc.next();
        auto lab = origin_edge_lengths[he_ab.edge().id.idx];
        auto lbc = origin_edge_lengths[he_bc.edge().id.idx];
        auto lca = origin_edge_lengths[he_ca.edge().id.idx];

        Vector2d pa{0.0, 0.0};
        Vector2d pb{lab, 0.0};
        Vector2d pc = triangle_oppo_point(lab, lbc, lca, false);
        double t = edge_points.back().t;
        if (he_ab.edge().halfedge().id != he_ab.id) {
            t = 1.0 - t;
        }
        Vector2d mid_pt{ pa * (1.0 - t) + pb * t };
        auto dir = Vector2d::Map(dir_data);
        Vector2d pd = mid_pt + dir;
        trace_next(mid_pt, pa, pb, pc, pd);
    }
}

template<std::size_t N, typename Mesh>
void TracePolyline<N, Mesh>::add_intersection_point(double left_ori, double right_ori, gpf::HalfedgeId hab) {
    using Vec = Eigen::Matrix<double, N, 1>;
    auto s = left_ori + right_ori;
    auto tb = left_ori / s;
    auto ta = 1.0 - tb;
    EdgePoint edge_point;
    auto he_ab = mesh->halfedge(hab);
    auto pa = Vec::Map(he_ab.from().prop().pt.data());
    auto pb = Vec::Map(he_ab.to().prop().pt.data());
    Vec::Map(edge_point.pt.data()) = pa * tb + pb * ta;
    auto e_ab = he_ab.edge();
    if (e_ab.halfedge().id == he_ab.id) {
        edge_point.t = ta;
    } else {
        edge_point.t = tb;
    }
    edge_point.eid = e_ab.id;
    auto pid = this->edge_points.size();
    this->edge_points.push_back(std::move(edge_point));
    this->path.emplace_back(pid);
    this->path_on_face_vec.emplace_back(he_ab.face().id);
}

}

template<typename Mesh>
void write_mesh1(const std::string& name, const Mesh& mesh) {
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

template<std::size_t N, typename VP, typename HP, typename EP, typename FP>
auto project_polylines_on_mesh(
    std::vector<std::array<double, 3>>& points,
    const std::vector<std::vector<std::size_t>>& polylines,
    gpf::ManifoldMesh<VP, HP, EP, FP>& mesh,
    std::unordered_map<gpf::FaceId, gpf::FaceId>* face_parent_map = nullptr,
    std::unordered_map<gpf::EdgeId, gpf::EdgeId>* edge_parent_map = nullptr
) {
    using Mesh = gpf::ManifoldMesh<VP, HP, EP, FP>;

    constexpr double EPS = 1e-3;
    auto point_vertices = detail::project_points_on_mesh(points, mesh, EPS, face_parent_map, edge_parent_map);
    detail::AuxiliaryMesh aux_mesh;
    aux_mesh.copy_from(mesh);
    gpf::update_edge_lengths<N>(aux_mesh, [mesh](auto v) {
        return std::span<const double, N>{mesh.vertex_prop(v.id).pt};
    });
    gpf::update_corner_angles(aux_mesh);
    gpf::update_vertex_angle_sums(aux_mesh);
    gpf::update_halfedge_signpost_angles(aux_mesh);
    gpf::update_halfedge_vectors(aux_mesh);

    std::vector<double> origin_signpost_angles(mesh.n_halfedges_capacity());
    for (auto he : aux_mesh.halfedges()) {
        origin_signpost_angles[he.id.idx] = he.prop().signpost_angle;
    }
    std::vector<double> origin_edge_lengths(mesh.n_edges_capacity());
    for (auto edge : aux_mesh.edges()) {
        origin_edge_lengths[edge.id.idx] = edge.prop().len;
    }

    detail::FlipGeodesic flip_geodesic{.mesh = &aux_mesh};
    std::vector<typename detail::TracePolyline<N, Mesh>::EdgePoint> edge_points;
    std::vector<std::vector<typename detail::TracePolyline<N, Mesh>::Anchor>> polyline_paths;
    std::vector<std::vector<gpf::FaceId>> path_segment_faces;
    polyline_paths.reserve(polylines.size());
    for (const auto polyline : polylines) {
        detail::TracePolyline<N, Mesh> trace {
            .origin_signpost_angles{origin_signpost_angles},
            .origin_edge_lengths{origin_edge_lengths},
            .edge_points{edge_points},
            .mesh{&mesh},
            .aux_mesh{&aux_mesh}
        };

        trace.path.push_back(point_vertices[polyline.front()]);
        for (std::size_t i = 0; i + 1 < polyline.size(); i++) {
            auto va = point_vertices[polyline[i]];
            auto vb = point_vertices[polyline[i + 1]];
            if (va.idx == 5735 && vb.idx == 5734) {
                const auto a = 2;
            }
            auto local_path =
                flip_geodesic.perform(detail::shortest_patch_by_dijksta(aux_mesh, va, vb,
                    [](auto e) { return e.prop().locked; },
                    [](auto e) { return e.prop().len; }));
            for (const auto hid : std::move(local_path)) {
                trace.trace_from_vertex(hid);
            }
        }
        polyline_paths.push_back(std::move(trace.path));
        path_segment_faces.push_back(std::move(trace.path_on_face_vec));
    }

    std::unordered_map<gpf::EdgeId, std::vector<std::size_t>> edge_to_points_map;
    std::vector<gpf::VertexId> edge_point_vertices(edge_points.size(), gpf::VertexId{});
    for (std::size_t pid = 0; pid < edge_points.size(); pid++) {
        const auto& point = edge_points[pid];
        if (point.t < EPS) {
            edge_point_vertices[pid] = mesh.edge(point.eid).halfedge().from().id;
        } else if (point.t > 1.0 - EPS) {
            edge_point_vertices[pid] = mesh.edge(point.eid).halfedge().to().id;
        } else {
            edge_to_points_map[point.eid].push_back(pid);
        }
    }

    std::unordered_map<gpf::FaceId, std::pair<detail::CCS3d, std::vector<typename detail::TracePolyline<N, Mesh>::Anchor>>> face_to_ccs_map;
    for (const auto& [path, faces] : std::views::zip(polyline_paths, path_segment_faces)) {
        for (std::size_t i = 0; i < faces.size(); i++) {
            const auto fid = faces[i];
            if (!fid.valid()) {
                continue;
            }
            const auto va = path[i];
            const auto vb = path[i + 1];
            auto iter = face_to_ccs_map.find(fid);
            if (iter != face_to_ccs_map.end()) {
                iter->second.second.push_back(va);
                iter->second.second.push_back(vb);
            } else {
                face_to_ccs_map.emplace(fid, std::make_pair(detail::CCS3d(mesh, fid), std::vector<typename detail::TracePolyline<N, Mesh>::Anchor>{va, vb}));
            }
        }
    }

    for (const auto& [eid, point_indices] : edge_to_points_map) {
        detail::split_edge_by_points(mesh, eid, [&edge_points](std::size_t pid) {
            return edge_points[pid].pt;
        }, point_indices, edge_point_vertices, EPS, edge_parent_map);
    }

    auto get_vertex_id = [&edge_point_vertices](auto anchor) {
        return std::visit([&edge_point_vertices](auto&& arg) -> gpf::VertexId {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, gpf::VertexId>) {
                return arg;
            } else {
                return edge_point_vertices[arg];
            }
        }, anchor);
    };

    for (const auto& [fid, ccs_and_segments] : face_to_ccs_map) {
        const auto& ccs = ccs_and_segments.first;
        const auto& segments = ccs_and_segments.second;
        auto segment_vertices = segments | std::views::transform(get_vertex_id) | std::ranges::to<std::vector>();
        detail::triangulate_on_face(mesh, fid, {}, ccs, {}, segment_vertices, edge_point_vertices, face_parent_map);
    }
    write_mesh1("after_triangulate.obj", mesh);

    return std::move(polyline_paths) | std::views::transform([&get_vertex_id, &mesh](auto&& path) {
        std::vector<gpf::HalfedgeId> halfedges;
        halfedges.reserve(path.size() - 1);
        for (std::size_t i = 0; i + 1 < path.size(); ++i) {
            const auto va = get_vertex_id(path[i]);
            const auto vb = get_vertex_id(path[i + 1]);
            if (va == vb) {
                continue;
            }
            auto hid = mesh.he_from_vertices(va, vb);
            if (hid.valid()) {
                halfedges.emplace_back(hid);
            } else {
                halfedges.append_range(detail::shortest_patch_by_dijksta(
                    mesh, va, vb,
                    [] (auto e) { return false; },
                    [] (auto e) {
                        auto [v1, v2] = e.vertices();
                        std::span<const double, N> p1{v1.prop().pt};
                        std::span<const double, N> p2{v2.prop().pt};
                        return std::sqrt(gpf::squared_distance(p1, p2));
                    }
                ));
            }
        }
        return halfedges;
    }) | std::ranges::to<std::vector>();
}
