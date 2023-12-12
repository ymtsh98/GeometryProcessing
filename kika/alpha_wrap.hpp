#pragma once

#pragma warning (push) 
#pragma warning (disable : 4819)
#pragma warning (disable : 4267)
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#pragma warning(pop)

#include <iostream>
#include <string>
#include <vector>

namespace kika::remesh {

	template <typename T = std::array<uint32_t, 3>>
	std::pair<Eigen::MatrixX3d, std::vector<std::array<uint32_t, 3>>> alphaWrap(
		const Eigen::MatrixX3d& pos, std::vector<T> tris, 
		const double relativeAlpha = 50., const double relativeOffset = 1000.) {

		using Point_3 = CGAL::Exact_predicates_inexact_constructions_kernel::Point_3;
		using Mesh = CGAL::Surface_mesh<Point_3>;

		std::vector<Point_3> points(pos.rows()); 
		for (size_t i = 0; i < points.size(); i++) points[i] = Point_3(pos(i, 0), pos(i, 1), pos(i, 2));

		Mesh mesh;

		CGAL::Polygon_mesh_processing::repair_polygon_soup(points, tris);
		CGAL::Polygon_mesh_processing::orient_polygon_soup(points, tris);
		CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, tris, mesh);
		CGAL::Bbox_3 bbox = CGAL::Polygon_mesh_processing::bbox(mesh);
		double diagLength = std::sqrt(
			CGAL::square(bbox.xmax() - bbox.xmin()) +
			CGAL::square(bbox.ymax() - bbox.ymin()) +
			CGAL::square(bbox.zmax() - bbox.zmin()));

		double alpha = diagLength / relativeAlpha;
		double offset = diagLength / relativeOffset;

		Mesh wrap;
		CGAL::alpha_wrap_3(mesh, alpha, offset, wrap);

		std::pair<Eigen::MatrixX3d, std::vector<std::array<uint32_t, 3>>> ret;
		ret.first.resize(wrap.number_of_vertices(), 3);
		int idx = 0;
		for (const auto& p : wrap.points()) {
			ret.first.row(idx++) << p.x(), p.y(), p.z();
		}

		ret.second.resize(wrap.number_of_faces());
		idx = 0;
		for (const auto& f : wrap.faces()) {
			int k = 0;
			for (const auto& v : wrap.vertices_around_face(wrap.halfedge(f))) {
				ret.second[idx][k++] = v.id();
			}
			idx++;
		}
	
		return ret;
	}

}