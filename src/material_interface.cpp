#include "material_interface.hpp"
#include "gpf/ids.hpp"
#include "gpf/mesh.hpp"
#include "gpf/surface_mesh.hpp"
#include "gpf/utils.hpp"
#include "tet_mesh.hpp"

#include <predicates/predicates.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <ranges>
#include <unordered_map>
#include <unordered_set>

#include <boost/functional/hash.hpp>
#include <boost/container/small_vector.hpp>

#include <fstream>
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
    boost::container::small_vector<std::size_t, 4> materials;

    std::array<gpf::VertexId, 2> parents;
    std::array<std::array<double, 2>, 2> vals;
    std::array<double, 2> ori{{0.0, 1.0}};
};

struct EdgeProp {
    boost::container::small_vector<std::size_t, 3> materials;
};

struct FaceProp {
    boost::container::small_vector<std::size_t, 2> materials;
    std::array<std::size_t, 2> cells{{0, gpf::kInvalidIndex}};
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
    void extract(ExtractInfo& info, const tet_mesh::TetMesh& tet_mesh, const tet_mesh::Tet& tet) noexcept;
private:
    double compute_vert_orientations(const gpf::VertexId vid);
    void split_edges() noexcept;
    void split_faces() noexcept;
    void split_cells() noexcept;
};

MaterialInterface::MaterialInterface() {
    mesh = Mesh::new_in(std::vector<std::vector<std::size_t>> {{2, 1, 3}, {0 ,2, 3}, {1, 0, 3}, {0, 1, 2}});
    for (auto v : mesh.vertices()) {
        auto& material = v.data().property.materials;
        materials.resize(4);
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
    for (auto f : mesh.faces()) {
        auto& prop = f.data().property;
        prop.materials[0] = f.id.idx;
        prop.materials[1] = 4;
        prop.pid = f.id.idx;
    }

    cells = { Cell{.faces{0, 2, 4, 6}} };
}

void MaterialInterface::add_material(const std::array<double, 4>& material) {
    auto mid = materials.size();
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

    mid += 4;
    if (n_neg == 0 || n_pos == 0) {
        if (n_zero >= 3) {
            for (auto face : mesh.faces()) {
                if (ranges::all_of(face.halfedges(), [] (auto he) { return he.to().data().property.ori[0] == 0.0;}) ) {
                    auto& f_prop = face.data().property;
                    assert(f_prop.materials[0] < 4);
                    f_prop.materials[0] = mid;
                    const auto pid = f_prop.pid;

                    for (auto he : face.halfedges()) {
                        auto& v_mats = he.to().data().property.materials;
                        auto& e_mats = he.edge().data().property.materials;
                        auto v_it = ranges::find_if(v_mats, [pid](std::size_t m) { return m == pid; });
                        if (v_it != ranges::end(v_mats)) {
                            *v_it = mid;
                        }
                        auto e_it = ranges::find_if(e_mats, [pid](std::size_t m) { return m == pid; });
                        if (e_it != ranges::end(e_mats)) {
                            *e_it = mid;
                        }
                    }

                    // Because the sorting is done before adding materials,
                    // the latest material cannot be overwhelming of the previous ones.
                    // So there is no cell generated for the latest material
                }
            }
        }
        return;
    }


    split_edges();
    split_faces();
    split_cells();
}
void MaterialInterface::extract(
    ExtractInfo& info,
    const tet_mesh::TetMesh& tet_mesh,
    const tet_mesh::Tet& tet
) noexcept {
    std::vector<bool> keep_faces(mesh.n_faces_capacity(), false);
    std::vector<std::size_t> point_indices(mesh.n_vertices_capacity(), gpf::kInvalidIndex);
    for (const auto face : mesh.faces()) {
        auto face_props = face.data().property;
        if (ranges::all_of(face_props.materials, [](auto m) { return m >= 4;})) {
            const auto pid = face_props.pid;
            if (pid >= 4 || ranges::all_of(face_props.materials, [this, pid](auto m) {
                return ranges::any_of(ranges::iota_view{1, 4}, [this, m, pid](auto i) { return std::abs(materials[m - 4][(i + pid) % 4]) > 1e-8;});
            })) {
                keep_faces[face.id.idx] = true;
                for (const auto he : face.halfedges()) {
                    auto vid = he.to().id.idx;
                    if (point_indices[vid] == gpf::kInvalidIndex) {
                        point_indices[he.to().id.idx] = 0;
                    }
                }
            }
        }
    }

    for (std::size_t vid = 0; vid < point_indices.size(); ++vid) {
        if (point_indices[vid] == gpf::kInvalidIndex) {
            continue;
        }
        if (vid < 4) {
            point_indices[vid] = info.add_vertex(tet.vertices[vid]);
        } else {

        }
    }

    using Mat = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    Mat V(point_indices.size(), 3);
    V.topRows(4) = Mat::ConstMapType(corners, 4, 3);

    for (std::size_t i = 4; i < point_indices.size(); ++i) {
        if (point_indices[i] != gpf::kInvalidIndex) {
            const auto& v_props = mesh.vertex_data(gpf::VertexId{i}).property;
            const auto [i1, i2] = v_props.parents;
            auto [a1, b1] = v_props.vals[0];
            auto [a2, b2] = v_props.vals[1];

            auto a1b2 = std::abs(a1) * std::abs(b2);
            auto a2b1 = std::abs(a2) * std::abs(b1);
            auto t = a1b2 / (a1b2 + a2b1);
            V.row(i) = V.row(i1.idx) * (1.0 - t) + V.row(i2.idx) * t;

            std::vector<double> vm(materials.size());
            for (std::size_t i = 0; i < materials.size(); i++) {
                vm[i] = materials[i][i1.idx] * (1.0 - t) + materials[i][i2.idx] * t;
            }
            if (std::abs(vm[0] - vm[1]) > 1e-6) {
                const auto a = 2;
            }
        }
    }

    for (std::size_t i = 0; i < point_indices.size(); i++) {
        if (point_indices[i] != gpf::kInvalidIndex) {
            info.points.emplace_back(std::array<double, 3>{{V(i, 0), V(i, 1), V(i, 2)}});
        }
    }
    for (const auto face : mesh.faces()) {
        if (keep_faces[face.id.idx]) {
            info.faces.emplace_back(face.halfedges() | views::transform([&point_indices](auto he) {
                return point_indices[he.to().id.idx];
            }) | ranges::to<std::vector>());
            auto& f = info.faces.back();
            if (f[0] == 18494 && f[1] == 18495 && f[2] == 18493) {
                const auto a = 2;
            }
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
    if (v_props.ori[0] == 0) {
        v_props.materials.emplace_back(this->materials.size() + 3);
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
        new_eh.data().property.materials = e_props.materials;
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
                        new_he.edge().data().property.materials = {{face.data().property.materials[0], face.data().property.materials[1], this->materials.size() + 3}};
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

void MaterialInterface::split_cells() noexcept {
    const auto mid = this->materials.size() + 3;
    const auto n_old_cells = cells.size();
    std::vector<std::size_t> neg_cell_faces;
    for(std::size_t cid = 0; cid < n_old_cells; ++cid) {
        std::vector<std::size_t> pos_cell_faces;

        gpf::VertexId start_zero_vid;
        std::size_t n_halfedges = 0;
        for (const auto ori_fid : cells[cid].faces) {
            auto fid = gpf::FaceId{gpf::strip_orientation(ori_fid)};
            auto face = mesh.face(fid);
            auto he = face.halfedge();
            auto va = he.from();
            auto vb = he.to();
            gpf::VertexId non_zero_vid;
            if (va.data().property.ori[0] == 0.0) {
                if (vb.data().property.ori[0] == 0.0) {
                    auto vc = he.next().to();
                    assert(vc.data().property.ori[0] != 0.0);

                    const bool is_pos = vc.data().property.ori[0] > 0.0;
                    if (is_pos) {
                        pos_cell_faces.emplace_back(ori_fid);
                        auto [v, h] = gpf::is_positive(ori_fid) ? std::pair{va, he} : std::pair(vb, he.twin());
                        v.data().halfedge = h.id;
                        if (!start_zero_vid.valid()) {
                            start_zero_vid = v.id;
                        }
                        n_halfedges += 1;
                    } else {
                        neg_cell_faces.emplace_back(ori_fid);
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
                    face.data().property.cells[ori_fid & 1] = n_old_cells;
                }
            }
        }

        if (n_halfedges < 3) {
            continue;
        }
        assert(start_zero_vid.valid());

        auto curr_vh = mesh.vertex(start_zero_vid);
        std::vector<gpf::HalfedgeId> new_halfedges;
        new_halfedges.reserve(n_halfedges);

        while(true) {
            auto he = curr_vh.halfedge();
            new_halfedges.emplace_back(he.id);
            curr_vh = he.to();
            if (curr_vh.id == start_zero_vid) {
                break;
            }
        }

        const auto new_fid = mesh.add_face_by_halfedges(new_halfedges, true);
        auto& new_face_props = mesh.face(new_fid).data().property;
        new_face_props.cells = {{n_old_cells, cid}};
        new_face_props.materials = {{ cells[cid].material, mid }};
        pos_cell_faces.emplace_back(gpf::oriented_index(new_fid.idx, true));
        neg_cell_faces.emplace_back(gpf::oriented_index(new_fid.idx, false));


        assert(neg_cell_faces.size() >= 4);
        assert(pos_cell_faces.size() >= 4);

        cells[cid].faces.swap(pos_cell_faces);
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

        auto mi = base_mi;
        mi.materials.reserve(tet_material_set.size());
        auto tet_material_indices = tet_material_set | ranges::to<std::vector>();
        auto materials = tet_material_indices | views::transform([&tvs, &tet_mesh](auto mid) {
            std::array<double, 4> material;
            for (auto [vid, v] : ranges::zip_view{tvs, material}) {
                v = tet_mesh.vertex(vid).data().property.distances[mid];
            }
            return material;
        }) | ranges::to<std::vector>();
        auto indices = ranges::iota_view{0ull, tet_material_indices.size()} | ranges::to<std::vector>();
        ranges::sort(indices, [&materials] (auto i1, auto i2) {
            return materials[i1] > materials[i2];
        });

        for (std::size_t i = 0; i < indices.size(); i++) {
            mi.add_material(materials[indices[i]]);
        }
        mi.extract(info, tet_mesh, tet);
    }
    write_mesh("output.obj", info.points, info.faces);
    const auto a = 2;
}
