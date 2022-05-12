#pragma once
#include <mygp/halfedge_mesh.hpp>
#include <mygp/geocore.hpp>

namespace gp::curv::impl {
	template <typename T>
	std::vector<size_t> rRingNeighbsCache(const heds::Mesh<T>& mesh, const size_t src, const double radius, 
		std::vector<double>& minDist, 
		std::vector<std::vector<uint32_t>>& oneringNeighbs) {
		std::vector<size_t> neighbs;

		using node = std::pair<double, size_t>;
		std::priority_queue<node, std::vector<node>, std::greater<node>> search;
		search.emplace(0.0, src);
		std::vector<bool> used(mesh.getNumVerts(), false);
		used[src] = true;
		minDist[src] = 0.0;

		while (!search.empty()) {
			auto [d, v] = search.top();
			search.pop();

			for (const auto& u : oneringNeighbs[v]) {
				double dist = d +(mesh.getPosition<Eigen::Vector3d>(u) - mesh.getPosition<Eigen::Vector3d>(v)).norm();
				if (dist < minDist[u] && dist < radius) {
					search.emplace(dist, u);
					minDist[u] = dist;
					if (used[u]) continue;
					neighbs.emplace_back(u);
					used[u] = true;
				}
			}
		}

		minDist[src] = std::numeric_limits<double>::max();
		for (const auto i : neighbs) minDist[i] = std::numeric_limits<double>::max();
		return neighbs;
	}

	template <typename T>
	std::vector<size_t> kRingNeighbsCache(const heds::Mesh<T>& mesh, const size_t src, const size_t k, 
		std::vector<std::vector<uint32_t>>& oneringNeighbs) {
		std::vector<size_t> neighbs;

		using node = std::pair<size_t, size_t>;
		std::priority_queue<node, std::vector<node>, std::greater<node>> search;
		search.emplace(0, src);
		std::vector<bool> used(mesh.getNumVerts(), false);
		used[src] = true;

		while (!search.empty()) {
			auto [d, v] = search.top();
			search.pop();

			if (d < k) {
				for (const auto& u : oneringNeighbs[v]) {
					if (!used[u]) {
						search.emplace(d + 1, u);
						neighbs.emplace_back(u);
						used[u] = true;
					}
				}
			}
		}

		return neighbs;
	}

	template <typename T>
	std::vector<size_t> decimateNeighbs(const heds::Mesh<T>& mesh, const size_t src, 
		const std::vector<size_t>& neighbs, const Eigen::MatrixXd& normals) {
		std::vector<size_t> decNeighbs;
		decNeighbs.reserve(neighbs.size());
		Eigen::Vector3d n = normals.row(src);
		for (const auto i : neighbs) {
			if (n.dot(normals.row(i)) > 0.0) decNeighbs.emplace_back(i);
		}
		return decNeighbs;
	}

	template <typename T>
	Eigen::Vector3d averageNormal(const heds::Mesh<T>& mesh, const size_t src, 
		const std::vector<size_t>& neighbs, const Eigen::MatrixXd& normals) {
		Eigen::Vector3d normal = normals.row(src);
		for (const auto i : neighbs) normal += normals.row(i);
		return normal.normalized();
	}

	template <typename T>
	Eigen::Matrix3d findLocalFrame(const heds::Mesh<T>& mesh, const size_t src, const Eigen::Vector3d& avgNormal) {
		Eigen::Vector3d e = mesh.getPosition<Eigen::Vector3d>(mesh.verts[src]->oneringNeighbs()[0]->id) -
			mesh.getPosition<Eigen::Vector3d>(src);
		Eigen::Vector3d lx = (e - e.dot(avgNormal) * avgNormal).normalized();
		Eigen::Vector3d ly = avgNormal.cross(lx).normalized();

		Eigen::Matrix3d lf;
		lf.row(0) = lx;
		lf.row(1) = ly;
		lf.row(2) = avgNormal;
		return lf;
	}

	template <typename T>
	Eigen::Vector<double, 5> quadricFitting(const heds::Mesh<T>& mesh, const size_t src,
		const std::vector<size_t>& neighbs, const Eigen::Matrix3d& localFrame) {
		Eigen::Vector<double, 5> quad;

		Eigen::Vector3d s = mesh.getPosition<Eigen::Vector3d>(src);
		Eigen::MatrixXd A(neighbs.size(), 5);
		Eigen::VectorXd b(neighbs.size());

		for (size_t j = 0; const auto i : neighbs) {
			Eigen::Vector3d k = mesh.getPosition<Eigen::Vector3d>(i) - s;
			double u = k.dot(localFrame.row(0));
			double v = k.dot(localFrame.row(1));
			double n = k.dot(localFrame.row(2));

			A(j, 0) = u * u;
			A(j, 1) = u * v;
			A(j, 2) = v * v;
			A(j, 3) = u;
			A(j, 4) = v;

			b(j) = n;
			j++;
		}

		quad = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
		return quad;
	}

	std::tuple<double, double, Eigen::Vector3d, Eigen::Vector3d>
		computeCurvature(const Eigen::Matrix3d& localFrame, const Eigen::Vector<double, 5>& quad) {
		Eigen::Vector3d n = Eigen::Vector3d(-quad[3], -quad[4], 1.0).normalized();

		double E = 1.0 + quad[3] * quad[3];
		double F = quad[3] * quad[4];
		double G = 1.0 + quad[4] * quad[4];
		double L = 2.0 * quad[0] * n[2];
		double M = quad[1] * n[2];
		double N = 2.0 * quad[2] * n[2];

		Eigen::Matrix2d S;
		S << L * G - M * F, M* E - L * F, M* E - L * F, N* E - M * F;
		S.array() /= E * G - F * F;

		Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(S, Eigen::ComputeEigenvectors);
		Eigen::Vector2d eval = solver.eigenvalues();
		Eigen::Matrix2d evec = solver.eigenvectors();

		eval = -eval;

		Eigen::Vector3d dir1, dir2;
		dir1 = localFrame.row(0) * evec(0, 0) + localFrame.row(1) * evec(1, 0);
		dir2 = localFrame.row(0) * evec(0, 1) + localFrame.row(1) * evec(1, 1);

		dir1 *= eval[0];
		dir2 *= eval[1];

		dir1.normalize();
		dir2.normalize();

		if (eval[0] > eval[1])
			return std::make_tuple(eval[1], eval[0], dir2, dir1);
		return std::make_tuple(eval[0], eval[1], dir1, dir2);
	}

}

namespace gp::curv {
	/// @param mesh 
	/// @param radius 
	/// @param kring 
	/// @return minimum/maximum curvature, minimum/maximum direction.
	template <typename T>
	std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd> 
		principalCurvature(const heds::Mesh<T>& mesh, const size_t radius=4, const bool kring=false) {
		size_t n = mesh.getNumVerts();
	
		Eigen::VectorXd minCurvature(n), maxCurvature(n);
		Eigen::MatrixXd minDirection(n, 3), maxDirection(n, 3);
	
		double mel = mesh.avgEdgeLength();
		Eigen::MatrixXd normals = util::vertexNormals(mesh);

		std::vector<double> cacheDist;
		if (!kring) cacheDist.resize(n, std::numeric_limits<double>::max());
		auto&& oneringNeighbs = mesh.adjacencyList();

		#pragma omp parallel
		{
			#pragma omp for firstprivate(cacheDist)
			for (int i = 0; i < n; i++) {
				std::vector<size_t> neighbs;
				if (!kring) neighbs = impl::rRingNeighbsCache(mesh, i, mel * radius, cacheDist, oneringNeighbs);
				if (kring || neighbs.size() < 6) neighbs = impl::kRingNeighbsCache(mesh, i, radius, oneringNeighbs);
				neighbs = impl::decimateNeighbs(mesh, i, neighbs, normals);

				auto normal = impl::averageNormal(mesh, i, neighbs, normals);
				auto lf = impl::findLocalFrame(mesh, i, normal);
				auto quad = impl::quadricFitting(mesh, i, neighbs, lf);
				auto [minc, maxc, mindir, maxdir] = impl::computeCurvature(lf, quad);

				minCurvature(i) = minc;
				maxCurvature(i) = maxc;
				minDirection.row(i) = mindir;
				maxDirection.row(i) = maxdir;

			}
			#pragma omp barrier
			{}
		}
		
		return std::make_tuple(minCurvature, maxCurvature, 
			minDirection, maxDirection);
	}
}