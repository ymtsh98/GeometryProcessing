#pragma once
#include <queue>
#include <mygp/geoutil.hpp>
#include <mygp/halfedge_mesh.hpp>
#include <mygp/union_find.hpp>

namespace gp::util {
	template <typename T1, typename T2>
	auto connectedMesh(const T1& pos, const T2& faces) {
		UnionFind uf(pos.size());
		for (const auto& face : faces) {
			for (size_t i = 1; i < face.size(); i++) uf.unite(face[0], face[i]);
		}

		std::unordered_map<size_t, std::vector<size_t>> s;
		for (size_t i = 0; i < pos.size(); i++) {
			s[uf.root(i)].emplace_back(i);
		}

		return s;
	}

	template <typename T>
	Eigen::MatrixXd vertexNormals(const heds::Mesh<T>& mesh) {
		Eigen::MatrixXd normals = Eigen::MatrixXd::Zero(mesh.getNumVerts(), 3);
		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			auto n = mesh.normal<Eigen::Vector3d>(mesh.faces[i]);
			for (const auto& v : mesh.faces[i]->adjacentVerts()) {
				normals.row(v->id) += n;
			}
		}
		normals.rowwise().normalize();
		return normals;
	}

	template <typename Derived, typename T=size_t, typename F = typename Derived::Scalar>
	auto vertexNormals(const Eigen::MatrixBase<Derived>& pos, const std::vector<T>& indices) {
		size_t n = (size_t)pos.rows();
	
		Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic> normals 
			= Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic>::Zero(n, 3);
	
		for (size_t i = 0; i < indices.size(); i+=3) {
			Eigen::Vector3<F> normal
				= cross3<Eigen::Vector3<F>>(pos.row(indices[i+1]) - pos.row(indices[i]), pos.row(indices[i+2]) - pos.row(indices[i]));
			
			normals.row(indices[i])   += normal;
			normals.row(indices[i+1]) += normal;
			normals.row(indices[i+2]) += normal;
		}
		normals.rowwise().normalize();
		return normals;
	}

	template <typename T1, typename T2=size_t, typename T3= typename T1::value_type>
	auto vertexNormals(const std::vector<T1>& pos, const std::vector<T2>& indices) {
		std::vector<T1> normals;
		if constexpr (IsEigen<T1>) normals.resize(pos.size(), T1::Zero(3));
		else normals.resize(pos.size(), T1{ (T3)0., (T3)0., (T3)0.});

		for (size_t i = 0; i < indices.size(); i+=3) {
			auto n = gp::util::cross3<std::vector<T3>>(pos[indices[i]], pos[indices[i+1]], pos[indices[i+2]]);
			for (size_t j = 0; j < 3; j++) {
				normals[indices[i]][j] += n[j];
				normals[indices[i+1]][j] += n[j];
				normals[indices[i+2]][j] += n[j];
			}
		}
		for (size_t i = 0; i < pos.size(); i++) {
			normals[i] = util::normalized(normals[i]);
		}
		return normals;
	}

	Eigen::Matrix4d lookAt(const Eigen::Vector3d& eye, const Eigen::Vector3d& center, const Eigen::Vector3d& up) {
		Eigen::Matrix4d matrix = Eigen::Matrix4d::Zero();
		Eigen::Vector3d t = eye - center;
		double l = t.norm();
		matrix(2)  = t[0] / l;
		matrix(6)  = t[1] / l;
		matrix(10) = t[2] / l;

		t[0] = up[1] * matrix(10) - up[2] * matrix(6);
		t[1] = up[2] * matrix(2)  - up[0] * matrix(10);
		t[2] = up[0] * matrix(6)  - up[1] * matrix(2);
		l = t.norm();
		matrix(0) = t[0] / l;
		matrix(4) = t[1] / l;
		matrix(8) = t[2] / l;

		matrix(1) = matrix(6) * matrix(8) - matrix(10)* matrix(4);
		matrix(5) = matrix(10) *matrix(0) - matrix(2) * matrix(8);
		matrix(9) = matrix(2) * matrix(4) - matrix(6) * matrix(0);

		matrix(12) = -(eye[0] * matrix(0) + eye[1] * matrix(4) + eye[2] * matrix(8));
		matrix(13) = -(eye[0] * matrix(1) + eye[1] * matrix(5) + eye[2] * matrix(9));
		matrix(14) = -(eye[0] * matrix(2) + eye[1] * matrix(6) + eye[2] * matrix(10));

		matrix(15) = 1.0;
		return matrix;
	}

	Eigen::Matrix4d perspective(const double fov, const double aspectRatio, const double zNear, const double zFar) {
		double yScale = 1.0 / tan(0.5 * fov);
		double xScale = yScale / aspectRatio;

		Eigen::Matrix4d matrix;
		matrix << xScale, 0, 0, 0,
			0, yScale, 0, 0,
			0, 0, -(zFar+zNear)/(zFar-zNear), -1,
			0, 0, -2*zNear*zFar/(zFar-zNear), 0;

		return matrix;
	}

	Eigen::MatrixXd vertexTexcoords(const Eigen::MatrixXd& pos, const double scale=0.01, 
		const Eigen::Vector3d& eye=Eigen::Vector3d::UnitX(), const Eigen::Vector3d& center = Eigen::Vector3d::Zero(), 
		const Eigen::Vector3d& up=Eigen::Vector3d::UnitY()) {
		Eigen::Matrix4d model = Eigen::Matrix4d::Identity();
		model(0, 0) = scale;
		model(1, 1) = scale;
		model(2, 2) = scale;
		auto view = lookAt(eye, center, up);
		auto proj = perspective(2.0*0.785398, 1.0, 0.01, 100.0);

		Eigen::Matrix4d mvp = model * view * proj;
		Eigen::MatrixXd uv(pos.rows(), 2);
		for (size_t i = 0; i < (size_t)pos.rows(); i++) {
			Eigen::Vector4d temp = mvp * 
				Eigen::Vector4d(pos(i, 0), pos(i, 1), pos(i, 2), 1.0);
			uv(i, 0) = temp(0);
			uv(i, 1) = temp(1);
		}
		return uv;
	}

	template <typename T>
	std::pair<Eigen::MatrixXd, std::vector<std::vector<int>>> dualMesh(const heds::Mesh<T>& mesh) {
		Eigen::MatrixXd verts(mesh.faces.size(), 3);
		for (size_t i = 0; i < mesh.getNumFaces(); i++) {
			verts.row(i) = mesh.center<Eigen::Vector3d>(mesh.faces[i]);
		}

		std::vector<std::vector<int>> faces;
		for (size_t i = 0; i < mesh.verts.size(); i++) {
			faces.emplace_back(std::vector<int>());
			const auto& fs = mesh.verts[i]->adjacentFaces();
			for (auto itr = fs.rbegin(); itr < fs.rend(); ++itr) {
				faces.back().emplace_back((int)(*itr)->id);
			}
		}

		return std::make_pair(verts, faces);
	}

	Eigen::VectorXd heatDiffusion(const Eigen::SparseMatrix<double>& lbo,
		const Eigen::VectorXd& areas,
		const Eigen::VectorXd& heat,
		const double time, 
		const size_t step = 1) {
		Eigen::VectorXd ret = heat;
		Eigen::SparseMatrix<double> A(lbo.rows(), lbo.rows());
		A.setIdentity();
		//A.diagonal().array() *= areas.array();

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A + time * lbo);

		for (size_t i = 0; i < step; i++) {
			ret = solver.solve(ret);
		}
		return ret;
	}

	template <typename Derived>
	void taubinSmoothing(Eigen::MatrixBase<Derived>& data, const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& areas, 
											 const size_t steps, const double shrink=0.33, const double inflate=0.34) {
		Eigen::VectorXd invA = 1.0/areas.array();
		Eigen::SparseMatrix<double> S = invA.asDiagonal() * lbo;
		invA = S.diagonal();
		invA = invA.asDiagonal().inverse().diagonal();
		S = invA.asDiagonal() * S;

		for (size_t i = 0; i < steps; i++) {
			data += inflate * S * data;
			data -= shrink * S * data;
		}
	}

	//template <typename T>
	//Eigen::SparseMatrix<double> gradient(const heds::Mesh<T>& mesh) {
	//	Eigen::SparseMatrix<double> grad(mesh.getNumFaces(), mesh.getNumVertices());
	//	std::vector<Eigen::Triplet<double>> triplets;
	//	
	//	for (size_t i = 0; i < mesh.getNumFaces(); i++) {
	//		auto normal = mesh.normal<Eigen::Vector3d>(mesh.faces[i]).normalized();
	//
	//		auto& v0 = mesh.faces[i]->edge->vert;
	//		auto& v1 = mesh.faces[i]->edge->next->vert;
	//		auto& v2 = mesh.faces[i]->edge->next->next->vert;
	//
	//		auto p0 = mesh.getPosition<Eigen::Vector3d>(v0->id);
	//		auto p1 = mesh.getPosition<Eigen::Vector3d>(v1->id);
	//		auto p2 = mesh.getPosition<Eigen::Vector3d>(v2->id);
	//
	//		triplets.emplace_back(i, v0->id, normal.cross(p2 - p1));
	//		triplets.emplace_back(i, v1->id, normal.cross(p0 - p2));
	//		triplets.emplace_back(i, v2->id, normal.cross(p1 - p0));
	//	}
	//	grad.setFromTriplets(triplets.begin(), triplets.end());
	//	return grad;
	//}

	template <typename T>
	std::vector<size_t> rRingNeighbs(const heds::Mesh<T>& mesh, const size_t src, const double radius) {
		std::vector<size_t> neighbs;

		using node = std::pair<double, size_t>;
		std::priority_queue<node, std::vector<node>, std::greater<node>> search;
		search.emplace(0.0, src);
		std::vector<std::pair<bool, double>> used(mesh.getNumVerts(), 
			std::make_pair(false, std::numeric_limits<double>::max()));
		used[src] = std::make_pair(true, 0.0);

		while (!search.empty()) {
			auto [d, v] = search.top();
			search.pop();

			for (const auto& u : mesh.verts[v]->oneringNeighbs()) {
				double dist = d + (mesh.getPosition<Eigen::Vector3d>(u->id) - mesh.getPosition<Eigen::Vector3d>(v)).norm();
				if (dist < used[u->id].second && dist < radius) {
					search.emplace(dist, u->id);
					used[u->id].second = dist;
					if (used[u->id].first) continue;
					neighbs.emplace_back(u->id);
					used[u->id].first = true;
				}
			}
		}

		return neighbs;
	}

	template <typename T>
	std::vector<size_t> kRingNeighbs(const heds::Mesh<T>& mesh, const size_t src, const size_t k) {
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
				for (const auto& u : mesh.verts[v]->oneringNeighbs()) {
					if (!used[u->id]) {
						search.emplace(d + 1, u->id);
						neighbs.emplace_back(u->id);
						used[u->id] = true;
					}
				}
			}
		}

		return neighbs;
	}

}

namespace gp::shape {
	template <typename T1, typename T2, typename T3=std::vector<double>>
	std::tuple<std::vector<std::vector<double>>,
		std::vector<uint32_t>,
		std::vector<std::vector<double>>,
		std::vector<std::vector<double>>> createSphere(const T1& center, const T2& radius, const size_t reso=10, const T3& color={ 0.7,0.7,0.7 }) {
		std::vector<std::vector<double>> sphere(reso*(reso-1) + 2, std::vector<double>(3));

		for (size_t i = 0; i < reso; i++) {
			for (size_t j = 0; j < reso-1; j++) {
				double s = 2.0 * i * std::numbers::pi / reso;
				double t = (j+1) * std::numbers::pi / reso;
				sphere[j * reso + i][0] = center[0] + radius * cos(s) * sin(t);
				sphere[j * reso + i][1] = center[1] + radius * sin(s) * sin(t);
				sphere[j * reso + i][2] = center[2] + radius * cos(t);
			}
		}

		sphere[sphere.size() - 2][0] = center[0];
		sphere[sphere.size() - 2][1] = center[1];
		sphere[sphere.size() - 2][2] = center[2] + radius;
		sphere[sphere.size() - 1][0] = center[0];
		sphere[sphere.size() - 1][1] = center[1];
		sphere[sphere.size() - 1][2] = center[2] - radius;

		std::vector<uint32_t> indices;
		for (size_t i = 0; i < reso; i++) {
			indices.emplace_back((uint32_t)(sphere.size()-2));
			indices.emplace_back((uint32_t)i);
			indices.emplace_back((uint32_t)((i+1)%reso));
			indices.emplace_back((uint32_t)(sphere.size()-1));
			indices.emplace_back((uint32_t)(reso*(reso-2)+(i+1)%reso));
			indices.emplace_back((uint32_t)(reso*(reso-2)+i));
		}
		for (size_t i = 0; i < reso-2; i++) {
			for (size_t j = 0; j < reso; j++) {
				indices.emplace_back((uint32_t)(reso*i+j));
				indices.emplace_back((uint32_t)(reso*i+j+reso));
				indices.emplace_back((uint32_t)(reso*i+(j+1)%reso+reso));
				indices.emplace_back((uint32_t)(reso*i+j));
				indices.emplace_back((uint32_t)(reso*i+(j+1)%reso+reso));
				indices.emplace_back((uint32_t)(reso*i+(j+1)%reso));
			}
		}

		auto normals = gp::util::vertexNormals(sphere, indices);
		std::vector<std::vector<double>> colors(sphere.size(),
																						std::vector<double>{(double)color[0], (double)color[1], (double)color[2]});
		return std::make_tuple(sphere, indices, normals, colors);
	}
}