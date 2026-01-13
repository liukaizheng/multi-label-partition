#include "material_interface.hpp"
#include "gpf/detail.hpp"
#include "gpf/ids.hpp"
#include "gpf/mesh.hpp"
#include "gpf/surface_mesh.hpp"
#include "gpf/utils.hpp"
#include "tet_mesh.hpp"

#include <bit>
#include <iterator>
#include <numeric>
#include <predicates/predicates.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <ranges>
#include <unordered_map>
#include <unordered_set>

#include <boost/functional/hash.hpp>
#include <boost/container/small_vector.hpp>

#include <fstream>
#include <utility>
auto write_mesh(const std::string& name, const std::vector<std::array<double, 3>>& uv, const std::vector<std::vector<std::size_t>>& F)
{
    std::ofstream file(name);
    for (Eigen::Index i = 0; i < uv.size(); ++i) {
        file << "v " << uv[i][0] << " " << uv[i][1] <<" " << uv[i][2] << "\n";
    }
    for (Eigen::Index i = 0; i < F.size(); ++i) {
        file << "f";
        for (Eigen::Index j = 0; j < F[i].size(); ++j) {
            file << "  " << F[i][j] + 1;
        }
        file << "\n";
    }
    file.close();
}

namespace ranges = std::ranges;
namespace views = std::views;

struct VertexProp {
    std::array<std::size_t, 4> materials;

    std::array<gpf::VertexId, 2> parents;
    std::array<std::array<double, 2>, 2> vals;
    std::array<double, 2> ori{{0.0, 1.0}};
};

struct EdgeProp {
    std::array<std::size_t, 3> materials;
    gpf::EdgeId parent;
};

struct FaceProp {
    std::array<std::size_t, 2> materials;
    std::array<std::size_t, 2> cells{{0, gpf::kInvalidIndex}};
    bool has_boundary{false};
};

struct Cell {
    std::vector<std::size_t> faces;
    std::size_t material{4ull};
};

using Mesh = gpf::SurfaceMesh<VertexProp, gpf::Empty, EdgeProp, FaceProp>;

struct EdgeVertexKeyHash {
    std::size_t operator()(const std::tuple<gpf::EdgeId, std::size_t, std::size_t>& key) const noexcept {
        std::size_t seed = 0;
        boost::hash_combine(seed, std::get<0>(key).idx);
        boost::hash_combine(seed, std::get<1>(key));
        boost::hash_combine(seed, std::get<2>(key));
        return seed;
    }
};

struct FaceVertexKeyHash {
    std::size_t operator()(const std::tuple<gpf::FaceId, std::size_t, std::size_t, std::size_t>& key) const noexcept {
        std::size_t seed = 0;
        boost::hash_combine(seed, std::get<0>(key).idx);
        boost::hash_combine(seed, std::get<1>(key));
        boost::hash_combine(seed, std::get<2>(key));
        boost::hash_combine(seed, std::get<3>(key));
        return seed;
    }
};

struct CellVertexKeyHash {
    std::size_t operator()(const std::array<std::size_t, 4>& key) const noexcept {
        return boost::hash_value(key);
    }
};

struct ExtractInfo {
    std::vector<std::array<double, 3>> points;
    std::vector<std::vector<std::size_t>> faces;
    std::unordered_map<gpf::VertexId, std::size_t> vertex_map;
    std::unordered_map<std::tuple<gpf::EdgeId, std::size_t, std::size_t>, std::size_t, EdgeVertexKeyHash> edge_vertex_map;
    std::unordered_map<std::tuple<gpf::FaceId, std::size_t, std::size_t, std::size_t>, std::size_t, FaceVertexKeyHash> face_vertex_map;
    std::unordered_map<std::array<std::size_t, 4>, std::size_t, CellVertexKeyHash> cell_vertex_map;

    std::size_t add_vertex(const gpf::VertexId vid) {
        return vertex_map.emplace(vid, points.size()).first->second;
    }

    std::size_t add_vertex(const gpf::EdgeId eid, const std::size_t m1, const std::size_t m2) {
        return edge_vertex_map.emplace(std::make_tuple(eid, m1, m2), points.size()).first->second;
    }

    std::size_t add_vertex(const gpf::FaceId fid, const std::size_t m1, const std::size_t m2, const std::size_t m3) {
        return face_vertex_map.emplace(std::make_tuple(fid, m1, m2, m3), points.size()).first->second;
    }

    std::size_t add_vertex(const std::size_t m1, const std::size_t m2, const std::size_t m3, const std::size_t m4) {
        return cell_vertex_map.emplace(std::array<std::size_t, 4>{{ m1, m2, m3, m4 }}, points.size()).first->second;
    }

};

struct MaterialInterface {
    Mesh mesh;
    std::vector<Cell> cells;
    std::vector<std::array<double, 4>> materials;

    MaterialInterface();
    MaterialInterface(const MaterialInterface&) = default;

    void add_material(const std::array<double, 4>& material);
    void extract(ExtractInfo& info, const tet_mesh::TetMesh& tet_mesh, const tet_mesh::Tet& tet, const std::vector<std::size_t>& material_indices) noexcept;
private:
    double compute_vert_orientations(const gpf::VertexId vid);
    void split_edges() noexcept;
    void split_faces() noexcept;
    void split_cells(const std::size_t max_v_idx) noexcept;
    void set_negative_face_properties(const gpf::FaceId fid, const std::size_t flag, const std::size_t new_cid, const std::size_t new_mid) noexcept;
    void merge_negative_cell_faces() noexcept;
};

void merge_collinear_edges_on_face(Mesh& mesh, auto& merge_halfedges_map, const gpf::FaceId fid) noexcept {
    auto face = mesh.face(fid);
    auto face_halfedges = face.halfedges();
    auto fh = *std::ranges::find_if(face_halfedges, [](const auto h) {
        auto prev = h.prev();
        return prev.edge().data().property.parent != h.edge().data().property.parent;
    });

    auto start_vid = fh.from().id;
    gpf::EdgeId parent;
    std::vector<gpf::HalfedgeId> halfedges;
    bool first = true;
    while(true) {
        auto curr_parent = fh.edge().data().property.parent;
        if (halfedges.empty()) {
            parent = curr_parent;
            halfedges.emplace_back(fh.id);
        } else {
            if (curr_parent == parent) {
                halfedges.emplace_back(fh.id);
            } else {
                assert(!halfedges.empty());
                if (halfedges.size() > 1) {
                    auto ha = mesh.halfedge(halfedges.front());
                    auto hb = mesh.halfedge(halfedges.back());
                    auto ha_prev = ha.prev();
                    auto va = ha_prev.to();
                    auto vb = hb.to();
                    // connect halfedges
                    ha_prev.data().next = hb.id;
                    hb.data().prev = ha_prev.id;
                    // set vertex halfedge
                    // [NOTICE]: there is no need to set vertex sibling, because they didn't be modified.
                    // There is also no need to delete unused verttices, because we will delete them when delete unused faces
                    va.data().halfedge = hb.id;
                    auto key = gpf::detail::ordered_pair(va.id, vb.id);
                    auto it = merge_halfedges_map.find(key);
                    gpf::EdgeId merged_eid{};
                    if (it == merge_halfedges_map.end()) {
                        merge_halfedges_map[key] = {{hb.id, hb.id }};
                        auto eb = hb.edge();
                        merged_eid = eb.id;
                        eb.data().halfedge = hb.id;
                    } else {
                        merged_eid = mesh.halfedge(it->second[0]).edge().id;
                        hb.data().edge = merged_eid;
                        mesh.halfedge(it->second[1]).data().sibling = hb.id;
                        it->second[1] = hb.id;
                    }
                    for (std::size_t i = 0; i + 1 < halfedges.size(); i++) {
                        auto he = mesh.halfedge(halfedges[i]);
                        mesh.delete_halfedge(he.id);
                        auto eid = he.edge().id;
                        if (eid != merged_eid) {
                            mesh.delete_edge(he.edge().id);
                        }
                    }
                }
                halfedges.clear();
                halfedges.emplace_back(fh.id);
                parent = curr_parent;
            }
        }
        if (!first && fh.from().id == start_vid) {
            break;
        }
        first = false;
        fh = fh.next();
    }
    face.data().halfedge = fh.id;
}

template<typename FaceRange, typename HashMap>
auto merge_faces(Mesh& mesh, HashMap& merge_halfedges_map, FaceRange&& face_range) noexcept {
    std::unordered_map<gpf::EdgeId, gpf::HalfedgeId> edge_halfedge_map;
    const auto new_fid = mesh.new_faces(1);
    bool first = true;
    for (const gpf::FaceId fid : std::forward<FaceRange>(face_range)) {
        auto face = mesh.face(fid);
        for (auto he : face.halfedges()) {
            auto it = edge_halfedge_map.emplace(he.edge().id, he.id);
            if (!it.second) {
                it.first->second = gpf::HalfedgeId{};
            }
        }
        if (first) {
            mesh.face(new_fid).data().property = face.data().property;
            first = false;
        }
        mesh.delete_face(fid);
    }

    gpf::HalfedgeId face_first_hid{};
    for (const auto [eid, hid] : edge_halfedge_map) {
        if (hid.valid()) {
            auto he = mesh.halfedge(hid);
            he.from().data().halfedge = he.id;
            if (!face_first_hid.valid()) {
                face_first_hid = hid;
            }
        }
        // Don't delete unused halfedges, `he.from()` maybe need them
        // We will delete them when deleting unused faces
    }

    gpf::HalfedgeId prev_hid{};
    mesh.face(new_fid).data().halfedge = face_first_hid;
    gpf::HalfedgeId curr_hid = face_first_hid;
    do {
        mesh.halfedge(curr_hid).data().face = new_fid;
        if (prev_hid.valid()) {
            mesh.connect_halfedges(prev_hid, curr_hid);
        }
        prev_hid = curr_hid;
        curr_hid = mesh.halfedge(curr_hid).to().halfedge().id;
    } while (curr_hid != face_first_hid);
    mesh.connect_halfedges(prev_hid, face_first_hid);

    merge_collinear_edges_on_face(mesh, merge_halfedges_map, new_fid);
    return new_fid;
}

gpf::HalfedgeId delete_sibling(Mesh& mesh, gpf::HalfedgeId first_hid) {
    auto get_valid_halfedge = [](auto he) {
        while(!he.data().vertex.valid()) {
            he = he.sibling();
        }
        return he;
    };
    auto curr_he = get_valid_halfedge(mesh.halfedge(first_hid));
    first_hid = curr_he.id;

    do {
        auto next_he = get_valid_halfedge(curr_he.sibling());
        curr_he.data().sibling = next_he.id;
        curr_he = std::move(next_he);
    } while (curr_he.id != first_hid);

    return first_hid;
}

gpf::HalfedgeId delete_incoming_next(Mesh& mesh, gpf::HalfedgeId first_hid) {
    auto get_valid_halfedge = [](auto he) {
        while(!he.data().vertex.valid()) {
            he = he.incoming_next();
        }
        return he;
    };
    auto curr_he = get_valid_halfedge(mesh.halfedge(first_hid));
    first_hid = curr_he.id;

    do {
        auto next_he = get_valid_halfedge(curr_he.incoming_next());
        curr_he.data().incoming_next = next_he.id;
        curr_he = std::move(next_he);
    } while (curr_he.id != first_hid);

    return first_hid;
}

void MaterialInterface::merge_negative_cell_faces() noexcept {
    std::array<std::vector<std::size_t>, 4> boundary_faces;
    std::unordered_map<gpf::FaceId, int> face_cancelling_map;
    for (const auto ori_fid : cells.back().faces) {
        const auto [fid_idx, reversed] = gpf::decode_index(ori_fid);
        auto fid = gpf::FaceId{fid_idx};
        const auto& f_props = mesh.face(fid).data().property;
        const auto pid = f_props.materials[0];
        if (pid < 4) {
            boundary_faces[pid].emplace_back(ori_fid);
        } else {
            face_cancelling_map[fid] += reversed ? -1 : 1;
        }
    }
    if (ranges::all_of(boundary_faces, [](const auto& faces) { return faces.size() <= 1; })) {
        return;
    }

    std::vector<std::size_t> new_cell_faces;
    new_cell_faces.reserve(cells.back().faces.size());
    std::unordered_map<std::pair<gpf::VertexId, gpf::VertexId>, std::array<gpf::HalfedgeId, 2>, gpf::detail::PairHash> merge_halfedges_map;
    for (const auto& ori_faces : boundary_faces) {
        if (ori_faces.size() == 1) {
            new_cell_faces.emplace_back(ori_faces[0]);
            mesh.reassign_face_vertex_halfedge(gpf::FaceId{gpf::strip_orientation(ori_faces[0])});
        } else if (ori_faces.size() > 1) {
            const auto fid = merge_faces(mesh, merge_halfedges_map, ori_faces | views::transform([](auto ori_fid) { return gpf::FaceId{gpf::strip_orientation(ori_fid)}; }));
            new_cell_faces.emplace_back(gpf::oriented_index(fid.idx, gpf::is_negative(ori_faces[0])));
        }
    }

    for (auto [ha, hb] : std::move(merge_halfedges_map) | views::values) {
        mesh.halfedge(hb).data().sibling = ha;
    }

    const auto check_vertex_is_deleted = [this](gpf::VertexId vid) {
        return ranges::all_of(mesh.vertex(vid).incoming_halfedges(), [this, vid](const auto& he) {
            auto v = he.to().id;
            return !v.valid() || v != vid;
        });
    };

    for (const auto [fid, cnt] : face_cancelling_map) {
        if (cnt != 0) {
            new_cell_faces.emplace_back(gpf::oriented_index(fid.idx, cnt < 0));
            mesh.reassign_face_vertex_halfedge(fid);
            continue;
        }
        auto face = mesh.face(fid);
        bool prev_edge_deleted = false;
        gpf::HalfedgeId prev_hid{};
        bool first_edge_deleted = false;
        gpf::VertexId prev_vid{};
        for (const auto he : face.halfedges()) {
            auto edge = he.edge();
            bool delete_edge = true;
            const auto curr_vid = he.to().id;
            for (const auto h : edge.halfedges()) {
                if (h.id == he.id || !h.face().data().halfedge.valid()) {
                    mesh.delete_halfedge(h.id);
                } else {
                    delete_edge = false;
                }
            }
            if (delete_edge) {
                mesh.delete_edge(edge.id);
                if (!prev_vid.valid()) {
                    first_edge_deleted = true;
                }
            } else {
                edge.data().halfedge = delete_sibling(mesh, he.id);
            }

            if (prev_edge_deleted && delete_edge && check_vertex_is_deleted(prev_vid)){
                mesh.delete_vertex(prev_vid);
            } else if (prev_vid.valid()) {
                delete_incoming_next(mesh, prev_hid);
            }

            prev_vid = curr_vid;
            prev_hid = he.id;
            prev_edge_deleted = delete_edge;
        }
        if (prev_edge_deleted && first_edge_deleted && check_vertex_is_deleted(prev_vid)) {
            mesh.delete_vertex(prev_vid);
        } else {
            delete_incoming_next(mesh, prev_hid);
        }
        mesh.delete_face(fid);
    }

    cells.back().faces.swap(new_cell_faces);

    #ifndef NDEBUG
    // Verify mesh validity after merging
    for (const auto ori_fid : cells.back().faces) {
        const auto [fid_idx, reversed] = gpf::decode_index(ori_fid);
        auto fid = gpf::FaceId{fid_idx};
        auto face = mesh.face(fid);

        // Verify face is not deleted and exists
        assert(face.data().halfedge.valid());

        // Verify cell-face reference consistency
        const auto& f_props = face.data().property;
        assert(f_props.cells[ori_fid & 1] == cells.size() - 1);

        // Verify halfedge connectivity for each face
        std::size_t edge_count = 0;
        for (auto he : face.halfedges()) {
            // Verify prev/next chain consistency
            assert(he.next().prev().id == he.id);
            assert(he.prev().next().id == he.id);

            // Verify halfedge belongs to this face
            assert(he.face().id == fid);

            // Verify vertex halfedge reference is valid
            auto v = he.from();
            assert(v.halfedge().id.valid());
            for (auto ih : v.incoming_halfedges()) {
                assert(ih.data().vertex.valid());
            }

            // Verify edge halfedges are valid
            auto edge = he.edge();
            assert(edge.halfedge().id.valid());
            for (auto eh : edge.halfedges()) {
                assert(eh.data().vertex.valid());
                assert(eh.edge().id == edge.id);
                assert(eh.face().data().halfedge.valid());
            }

            edge_count++;
        }

        // Verify face has at least 3 edges (valid polygon)
        assert(edge_count >= 3);
    }
    #endif
}

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
        auto& e_props = e.data().property;
        e_props.parent = e.id;
        auto& material = e_props.materials;
        for (auto [m, fid] : ranges::zip_view{material, e.halfedges() | views::transform([](auto h) { return h.face().id.idx; })}) {
            m = fid;
        }
        material.back() = 4;
    }
    for (auto f : mesh.faces()) {
        auto& prop = f.data().property;
        prop.materials[0] = f.id.idx;
        prop.materials[1] = 4;
    }

    cells = { Cell{.faces{0, 2, 4, 6}} };
}

void MaterialInterface::add_material(const std::array<double, 4>& material) {
    materials.emplace_back(material);
    if (materials.size() == 1) {
        return;
    }

    int n_pos = 0;
    int n_neg = 0;
    int n_zero = 0;
    for (const auto v :mesh.vertices()) {
        double ori = compute_vert_orientations(v.id);
        if (ori > 0) {
            n_pos++;
        } else if (ori < 0) {
            n_neg++;
        } else {
            n_zero++;
        }
    }

    if (n_zero < 3 && n_neg == 0) {
        return;
    }

    const auto max_v_idx = mesh.n_vertices_capacity();

    split_edges();
    split_faces();
    split_cells(max_v_idx);
    auto face_vertices = mesh.faces() |
        views::transform([](const auto& face) {
            return views::transform(face.halfedges(), [](const auto& he) {
                return he.to().id.idx;
            });
        }) | ranges::to<std::vector<std::vector<std::size_t>>>();
    merge_negative_cell_faces();
}
void MaterialInterface::extract(
    ExtractInfo& info,
    const tet_mesh::TetMesh& tet_mesh,
    const tet_mesh::Tet& tet,
    const std::vector<std::size_t>& material_indices
) noexcept {
    const auto n_old_global_points = info.points.size();
    std::unordered_set<gpf::FaceId> kept_faces;
    std::vector<std::size_t> point_indices(mesh.n_vertices_capacity(), gpf::kInvalidIndex);
    for (const auto face : mesh.faces()) {
        const auto& face_props = face.data().property;
        bool keep = false;
        if (face.id.idx >= 4) {
            keep = face_props.materials[0] >= 4 && face_props.materials[1] >= 4;
        } else if (face_props.has_boundary) {
            const auto pid = face_props.materials[0];
            if(cells[face_props.cells[0]].material == face_props.materials[1]) {
                const auto& mat = materials[face_props.materials[1] - 4];
                keep = ranges::any_of(ranges::iota_view{1, 4}, [pid, &mat](auto i) { return std::abs(mat[(i + pid) % 4]) > 1e-8;});
            }
        }

        if (keep) {
            kept_faces.emplace(face.id);
            for (const auto he : face.halfedges()) {
                auto vid = he.to().id.idx;
                if (point_indices[vid] == gpf::kInvalidIndex) {
                    point_indices[he.to().id.idx] = 0;
                }
            }
        }
    }

    if (kept_faces.empty()) {
        return;
    }

    using Mat = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    Mat V(point_indices.size(), 3);

    #ifndef NDEBUG
    Eigen::MatrixXd VM(point_indices.size(), material_indices.size());
    #endif
    for (std::size_t vid = 0; vid < point_indices.size(); ++vid) {
        if (vid < 4) {
            V.row(vid) = Eigen::RowVector3d::ConstMapType(tet_mesh.vertex(tet.vertices[vid]).data().property.pt.data());
            if (point_indices[vid] != gpf::kInvalidIndex) {
                point_indices[vid] = info.add_vertex(tet.vertices[vid]);
                if (point_indices[vid] == info.points.size()) {
                    info.points.emplace_back(std::array<double, 3>{{V(vid, 0), V(vid, 1), V(vid, 2)}});
                }

            }
            #ifndef NDEBUG
            for (std::size_t i = 0; i < materials.size(); ++i) {
                VM(vid, i) = materials[i][vid];
            }
            #endif
        } else if (point_indices[vid] != gpf::kInvalidIndex) {
            const auto& vm = mesh.vertex(gpf::VertexId{vid}).data().property.materials; // vertex materials
            const auto count = ranges::distance(ranges::begin(vm), ranges::find_if(vm, [](const auto m) {return m >= 4; }));
            assert(count <= 2);
            switch(count) {
            case 0:
                point_indices[vid] = info.add_vertex(material_indices[vm[0] - 4], material_indices[vm[1] - 4], material_indices[vm[2] - 4], material_indices[vm[3] - 4]);
                break;
            case 1:
                point_indices[vid] = info.add_vertex(tet.faces[vm[0]], material_indices[vm[1] - 4], material_indices[vm[2] - 4], material_indices[vm[3] - 4]);
                break;
            case 2:
                point_indices[vid] = info.add_vertex(tet.edges[tet_mesh::Tet::edge_index(vm[0], vm[1])], material_indices[vm[2] - 4], material_indices[vm[3] - 4]);
                break;
            }
            if (point_indices[vid] != info.points.size()) {
                V.row(vid) = Eigen::RowVector3d::ConstMapType(info.points[point_indices[vid]].data());

                #ifndef NDEBUG
                const auto& v_props = mesh.vertex_data(gpf::VertexId{vid}).property;
                const auto [i1, i2] = v_props.parents;
                auto [a1, b1] = v_props.vals[0];
                auto [a2, b2] = v_props.vals[1];

                auto a1b2 = std::abs(a1) * std::abs(b2);
                auto a2b1 = std::abs(a2) * std::abs(b1);
                auto t = a1b2 / (a1b2 + a2b1);
                assert( ((V.row(i1.idx) * (1.0 - t) + V.row(i2.idx) * t) - V.row(vid)).norm() < 1e-8);

                VM.row(vid) = VM.row(i1.idx) * (1.0 - t) + VM.row(i2.idx) * t;
                auto indices = v_props.materials |
                    views::filter([] (auto mid) { return mid >= 4; }) |
                    views::transform([] (auto mid) { return mid - 4; }) |
                    ranges::to<std::vector>();
                for (std::size_t i = 1; i < indices.size(); ++i) {
                    assert(std::abs(VM(vid, indices[i]) - VM(vid, indices[i - 1])) < 1e-8);
                }
                #endif
            } else {
                const auto& v_props = mesh.vertex_data(gpf::VertexId{vid}).property;
                const auto [i1, i2] = v_props.parents;
                auto [a1, b1] = v_props.vals[0];
                auto [a2, b2] = v_props.vals[1];

                auto a1b2 = std::abs(a1) * std::abs(b2);
                auto a2b1 = std::abs(a2) * std::abs(b1);
                auto t = a1b2 / (a1b2 + a2b1);
                V.row(vid) = V.row(i1.idx) * (1.0 - t) + V.row(i2.idx) * t;
                info.points.emplace_back(std::array<double, 3>{{V(vid, 0), V(vid, 1), V(vid, 2)}});

                #ifndef NDEBUG
                VM.row(vid) = VM.row(i1.idx) * (1.0 - t) + VM.row(i2.idx) * t;
                auto indices = v_props.materials |
                    views::filter([] (auto mid) { return mid >= 4; }) |
                    views::transform([] (auto mid) { return mid - 4; }) |
                    ranges::to<std::vector>();
                for (std::size_t i = 1; i < indices.size(); ++i) {
                    assert(std::abs(VM(vid, indices[i]) - VM(vid, indices[i - 1])) < 1e-8);
                }
                #endif
            }
        } else {
            const auto& v_props = mesh.vertex_data(gpf::VertexId{vid}).property;
            const auto [i1, i2] = v_props.parents;
            auto [a1, b1] = v_props.vals[0];
            auto [a2, b2] = v_props.vals[1];

            auto a1b2 = std::abs(a1) * std::abs(b2);
            auto a2b1 = std::abs(a2) * std::abs(b1);
            auto t = a1b2 / (a1b2 + a2b1);
            V.row(vid) = V.row(i1.idx) * (1.0 - t) + V.row(i2.idx) * t;
            #ifndef NDEBUG
            VM.row(vid) = VM.row(i1.idx) * (1.0 - t) + VM.row(i2.idx) * t;
            #endif
        }
    }

    for (const auto fid : kept_faces) {
        const auto face = mesh.face(fid);
        const auto& f_props = face.data().property;
        const auto& f_materials = f_props.materials;
        if (f_materials[0] < 4) {
            info.faces.emplace_back(face.halfedges() | views::transform([&point_indices](auto he) {
                return point_indices[he.to().id.idx];
            }) | ranges::to<std::vector>());
        } else {
            assert(cells[f_props.cells[0]].material > cells[f_props.cells[1]].material);
            info.faces.emplace_back(face.halfedges_reverse() | views::transform([&point_indices](auto he) {
                return point_indices[he.to().id.idx];
            }) | ranges::to<std::vector>());
        }
    }
}

double MaterialInterface::compute_vert_orientations(const gpf::VertexId vid) {
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
    auto& v_props = mesh.vertex(vid).data().property;
    for (const auto i : v_props.materials) {
        if (i < 4) {
            vertex_flags |= (1 << i);
        } else {
            material_indices.push_back(i - 4);
        }
    }

    auto compute_sign_0 = [this, &material_indices](const std::size_t i) {
        const auto v0 = materials.back()[i];
        const auto v = materials[material_indices[0]][i];
        return predicates::mi_orient0d(v0, v);
    };

    auto compute_sign_1 = [this, &material_indices](const std::size_t i, const std::size_t j) {
        const auto& m1 = materials[material_indices[0]];
        const auto& m2 = materials[material_indices[1]];
        const auto& m = materials.back();
        double tm1[2] = {m1[i], m1[j]};
        double tm2[2] = {m2[i], m2[j]};
        double tm[2] = {m[i], m[j]};
        return predicates::mi_orient1d(tm1, tm2, tm);
    };

    auto compute_sign_2 = [this, &material_indices](const std::size_t i, const std::size_t j, const std::size_t k) {
        const auto& m1 = materials[material_indices[0]];
        const auto& m2 = materials[material_indices[1]];
        const auto& m3 = materials[material_indices[2]];
        const auto& m = materials.back();
        double tm1[3] = {m1[i], m1[j], m1[k]};
        double tm2[3] = {m2[i], m2[j], m2[k]};
        double tm3[3] = {m3[i], m3[j], m3[k]};
        double tm[3] = {m[i], m[j], m[k]};
        return predicates::mi_orient2d(tm1, tm2, tm3, tm);
    };

    auto compute_sign_3 = [this, &material_indices]() {
        const auto& m1 = materials[material_indices[0]];
        const auto& m2 = materials[material_indices[1]];
        const auto& m3 = materials[material_indices[2]];
        const auto& m4 = materials[material_indices[3]];
        const auto& m = materials.back();
        return predicates::mi_orient4d(m1.data(), m2.data(), m3.data(), m4.data(), m.data());
    };

    const auto& shape = M[vertex_flags];
    if (shape[0] == 1) {
        v_props.ori = compute_sign_0(shape[1]);
    } else if (shape[0] == 2) {
        v_props.ori = compute_sign_1(shape[1], shape[2]);
    } else if (shape[0] == 3) {
        v_props.ori = compute_sign_2(shape[1], shape[2], shape[3]);
    } else if (shape[0] == 4) {
        v_props.ori = compute_sign_3();
    }
    return v_props.ori[0];
}

void MaterialInterface::split_edges() noexcept {
    const auto old_edges = mesh.edges() | views::transform([] (auto e) { return e.id; }) | ranges::to<std::vector>();
    for (const auto eid : old_edges) {
        std::array<gpf::VertexId, 2> verts;
        std::array<std::array<double, 2>, 2> oris;
        for (auto [v, o, vh] : views::zip(verts, oris, mesh.edge(eid).vertices())) {
            v = vh.id;
            o = vh.data().property.ori;
        }
        if (oris[0][0] * oris[1][0] >= 0) {
            continue;
        }
        auto new_vid = mesh.split_edge(eid);
        auto new_vh = mesh.vertex(new_vid);
        const auto& e_props = mesh.edge_data(eid).property;
        auto new_eh = new_vh.halfedge().prev().edge();
        new_eh.data().property = e_props;
        auto& new_v_props = new_vh.data().property;
        auto& new_mat = new_v_props.materials;
        new_mat[0] = e_props.materials[0];
        new_mat[1] = e_props.materials[1];
        new_mat[2] = e_props.materials[2];
        new_mat[3] = this->materials.size() + 3;
        new_v_props.parents.swap(verts);
        new_v_props.vals.swap(oris);
    }
}

void MaterialInterface::split_faces() noexcept {
    auto old_faces = mesh.faces() | views::transform([] (auto f) { return f.id; }) | ranges::to<std::vector>();
    for (const auto fid : old_faces) {
        auto face = mesh.face(fid);
        auto curr_he = face.halfedge();
        gpf::VertexId first_zero_vid;
        auto first_hid = curr_he.id;

        auto prev_vh = curr_he.from();
        auto prev_ori = prev_vh.data().property.ori[0];

        if (prev_ori == 0.0) {
            first_zero_vid = prev_vh.id;
        }
        while(true) {
            auto vh = curr_he.to();
            auto ori = vh.data().property.ori[0];
            auto next_he = curr_he.next();
            if (ori == 0.0) {
                if (prev_ori == 0.0) {
                    face.data().halfedge = curr_he.id;
                    break;
                } else {
                    if (first_zero_vid.valid()) {
                        if (next_he.to().id == first_zero_vid) {
                            face.data().halfedge = next_he.id;
                            break;
                        } else if (vh.id == first_zero_vid) {
                            break;
                        }
                        auto new_he = mesh.halfedge(mesh.split_face(fid, first_zero_vid, vh.id));
                        auto new_fh = new_he.face();
                        face = mesh.face(fid); // new face added, rebind face
                        new_fh.data().property = face.data().property;
                        auto& new_e_props = new_he.edge().data().property;
                        new_e_props.materials = {{face.data().property.materials[0], face.data().property.materials[1], this->materials.size() + 3}};
                        new_e_props.parent = new_he.data().edge;
                        for (std::size_t i = 0; i < 2; i++) {
                            auto cid = face.data().property.cells[i];
                            if (cid != gpf::kInvalidIndex) {
                                cells[cid].faces.emplace_back((new_fh.id.idx << 1) + i);
                            }
                        }
                        break;
                    } else {
                        first_zero_vid = vh.id;
                    }
                }
            }
            curr_he = next_he;
            if (curr_he.id == first_hid) {
                break;
            }
            prev_ori = ori;
        }
    }
}
void MaterialInterface::set_negative_face_properties(
    const gpf::FaceId fid,
    const std::size_t flag,
    const std::size_t new_cid,
    const std::size_t new_mid
) noexcept {
    auto face = mesh.face(fid);
    auto& f_props = face.data().property;
    f_props.cells[flag] = new_cid;
    f_props.materials[1] = new_mid;

    auto& mat = f_props.materials;

    for (auto he : face.halfedges()) {
        he.to().data().property.materials[3] = new_mid;
        he.edge().data().property.materials[2] = new_mid;
    }
}

void MaterialInterface::split_cells(const std::size_t max_v_idx) noexcept {
    const auto mid = this->materials.size() + 3;
    const auto n_old_cells = cells.size();
    std::vector<std::size_t> neg_cell_faces;
    gpf::FaceId new_fid{};
    for(std::size_t cid = 0; cid < n_old_cells; ++cid) {
        std::vector<std::size_t> pos_cell_faces;

        std::size_t coplanar_ori_fid = gpf::kInvalidIndex;
        gpf::VertexId start_zero_vid{};
        std::size_t n_halfedges = 0;
        for (const auto ori_fid : cells[cid].faces) {
            auto fid = gpf::FaceId{gpf::strip_orientation(ori_fid)};
            auto face = mesh.face(fid);
            assert(face.data().property.cells[ori_fid & 1] == cid);
            auto he = face.halfedge();
            auto va = he.from();
            auto vb = he.to();
            gpf::VertexId non_zero_vid;
            if (va.data().property.ori[0] == 0.0) {
                if (vb.data().property.ori[0] == 0.0) {
                    auto vc = he.next().to();
                    const auto vc_ori = vc.data().property.ori[0];
                    if (vc_ori == 0.0) {
                        assert(coplanar_ori_fid == gpf::kInvalidIndex);
                        coplanar_ori_fid = ori_fid;
                        continue;
                    }

                    const bool is_pos = vc.data().property.ori[0] > 0.0;
                    if (is_pos) {
                        pos_cell_faces.emplace_back(ori_fid);
                    } else {
                        neg_cell_faces.emplace_back(ori_fid);
                        set_negative_face_properties(fid, ori_fid & 1, n_old_cells, mid);
                        auto [v, h] = gpf::is_positive(ori_fid) ? std::pair(vb, he.twin()) : std::pair{va, he};
                        face.data().property.cells[ori_fid & 1] = n_old_cells;
                        v.data().halfedge = h.id;
                        if (!start_zero_vid.valid()) {
                            start_zero_vid = v.id;
                        }
                        n_halfedges += 1;
                    }
                } else {
                    non_zero_vid = vb.id;
                }
            } else {
                non_zero_vid = va.id;
            }

            if (non_zero_vid.valid()) {
                if(mesh.vertex(non_zero_vid).data().property.ori[0] > 0.0) {
                    pos_cell_faces.push_back(ori_fid);
                } else {
                    neg_cell_faces.push_back(ori_fid);
                    set_negative_face_properties(fid, ori_fid & 1, n_old_cells, mid);
                }
            }
        }

        if (coplanar_ori_fid == gpf::kInvalidIndex && n_halfedges < 3) {
            continue;
        }
        assert(coplanar_ori_fid != gpf::kInvalidIndex || start_zero_vid.valid());

        auto curr_vh = mesh.vertex(start_zero_vid);
        std::vector<gpf::HalfedgeId> new_halfedges;
        new_halfedges.reserve(n_halfedges);

        while(coplanar_ori_fid == gpf::kInvalidIndex) {
            auto he = curr_vh.halfedge();
            new_halfedges.emplace_back(he.id);
            curr_vh = he.to();
            if (curr_vh.id == start_zero_vid) {
                break;
            }
        }
        if (coplanar_ori_fid != gpf::kInvalidIndex) {
            assert(pos_cell_faces.size() + 1 == cells[cid].faces.size() || pos_cell_faces.empty());
            new_fid.idx = gpf::strip_orientation(coplanar_ori_fid);
            if (pos_cell_faces.empty()) {
                auto face = mesh.face(new_fid);
                auto& f_props = face.data().property;
                assert(f_props.cells[coplanar_ori_fid & 1] == cid);
                f_props.cells[coplanar_ori_fid & 1] = n_old_cells;
                f_props.has_boundary = true;
                cells[cid].faces.clear();
                neg_cell_faces.emplace_back(coplanar_ori_fid);
                f_props.materials[1] = mid;
            } else {
                mesh.face(new_fid).data().property.has_boundary = true;
            }
        } else {
            new_fid = mesh.add_face_by_halfedges(new_halfedges, true);
            auto& new_face_props = mesh.face(new_fid).data().property;
            new_face_props.cells = {{n_old_cells, cid}};
            new_face_props.materials = {{ cells[cid].material, mid }};
            pos_cell_faces.emplace_back(gpf::oriented_index(new_fid.idx, true));
            neg_cell_faces.emplace_back(gpf::oriented_index(new_fid.idx, false));

            assert(neg_cell_faces.size() >= 4);
            assert(pos_cell_faces.size() >= 4);

            cells[cid].faces.swap(pos_cell_faces);
        }

    }
    if (!new_fid.valid()) {
        assert(neg_cell_faces.size() == std::transform_reduce(cells.begin(), cells.end(), std::size_t{0}, std::plus{}, [](const auto& cell) { return cell.faces.size(); }));
        for (auto& cell : cells) {
            cell.faces.clear();
        }
    }
    cells.emplace_back(Cell{.faces{std::move(neg_cell_faces)}, .material = mid });
}

void save_tet(
    const std::size_t tid,
    const std::vector<tet_mesh::Tet> &tets,
    const tet_mesh::TetMesh &tet_mesh
) {
    std::ofstream out("tet_" + std::to_string(tid) + ".obj");
    const auto& tvs = tets[tid].vertices;
    for (const auto vid : tvs) {
        const auto& pt = tet_mesh.vertex(vid).data().property.pt;
        out << "v " << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
    }
    out << "f 1 2 3\n";
    out << "f 1 4 2\n";
    out << "f 2 4 3\n";
    out << "f 3 4 1\n";
    out.close();
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

    ExtractInfo info;
    std::array<double, 12> corners;
    std::size_t tid = 0;
    for (const auto &tet : tets) {
        tid += 1;
        std::unordered_set<std::size_t> tet_material_set;
        for (const auto vid : tet.vertices) {
            const auto idx = vid.idx;
            tet_material_set.insert(v_high_materials.begin() + separators[idx], v_high_materials.begin() + separators[idx + 1]);
        }
        if (tet_material_set.size() < 2) {
            continue;
        }

        const auto& tvs = tet.vertices;
        for (std::size_t i = 0; i < 4; i++) {
            auto p1 = &corners[i * 3];
            const auto& p2 = tet_mesh.vertex_data(tvs[i]).property.pt;
            p1[0] = p2[0];
            p1[1] = p2[1];
            p1[2] = p2[2];
        }

        std::array<double, 4> min_material_values;
        for (auto [vid, v] : ranges::zip_view{tvs, min_material_values}) {
            auto& vals = tet_mesh.vertex(vid).data().property.distances;
            v = ranges::min(tet_material_set | views::transform([&vals, vid] (auto m) { return vals[m]; }));
        }
        for (std::size_t m = 0; m < n_materials; m++) {
            if (tet_material_set.contains(m)) {
                continue;
            }
            auto count = ranges::count_if(ranges::zip_view{tvs, std::move(min_material_values)},  [&tet_mesh, m] (auto pair) {
                auto [vid, v] = pair;
                return tet_mesh.vertex(vid).data().property.distances[m] > v;
            });

            if (count > 1) {
                tet_material_set.insert(m);
            }
        }
        if (tvs[0].idx == 33557 && tvs[1].idx == 42399 && tvs[2].idx == 42401 && tvs[3].idx == 16516) {
            const auto a = 2;
        }

        auto mi = base_mi;
        mi.materials.reserve(tet_material_set.size());
        auto tet_material_indices = tet_material_set | ranges::to<std::vector>();
        ranges::sort(tet_material_indices);
        auto materials = tet_material_indices | views::transform([&tvs, &tet_mesh](auto mid) {
            std::array<double, 4> material;
            for (auto [vid, v] : ranges::zip_view{tvs, material}) {
                v = tet_mesh.vertex(vid).data().property.distances[mid];
            }
            return material;
        }) | ranges::to<std::vector>();

        std::vector<std::array<std::size_t, 4>> material_int_values(materials.size());
        for (std::size_t i = 0; i < materials.size(); ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                material_int_values[i][j] = std::bit_cast<std::size_t>(materials[i][j]);
            }
        }

        for (const auto& m : materials) {
            mi.add_material(m);
        }

        mi.extract(info, tet_mesh, tet, tet_material_indices);
    }
    write_mesh("output.obj", info.points, info.faces);
    const auto a = 2;
}
