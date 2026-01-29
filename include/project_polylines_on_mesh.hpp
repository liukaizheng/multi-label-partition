#include <vector>
#include <array>
#include <gpf/manifold_mesh.hpp>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>


namespace detail {
template<typename VP, typename HP, typename EP, typename FP>
void project_points_on_mesh(
    const std::vector<std::array<double, 3>>& points,
    gpf::ManifoldMesh<VP, HP, EP, FP>& mesh
) {
    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Point_3 = Kernel::Point_3;
    using Triangle_3 = Kernel::Triangle_3;
    using TreeIterator = std::vector<Triangle_3>::const_iterator;
    using TreePrimitive = CGAL::AABB_triangle_primitive_3<Kernel, TreeIterator>;
    using TreeTraits = CGAL::AABB_traits_3<Kernel, Triangle_3>;
    using Tree = CGAL::AABB_tree<TreeTraits>;

    std::vector<Triangle_3> triangles;
}
}

template<typename VP, typename HP, typename EP, typename FP>
auto project_polylines_on_mesh(
    const std::vector<std::array<double, 3>>& points,
    const std::vector<std::vector<std::size_t>>& polylines,
    gpf::ManifoldMesh<VP, HP, EP, FP>& mesh
) {

}
