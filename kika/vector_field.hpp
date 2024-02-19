#pragma once

#include <kika/kikautil.hpp>
#include <kika/eigutil.hpp>
#include <kika/surface_mesh.hpp>

namespace kika::vf {

enum class Form {
	Vwise,
	Ewise,
	Fwise
};

enum class Space {
	Intrinsic,
	Extrinsic
};

using enum Form;
using enum Space;

class VectorField {
	Form form;
	Space space;
	kika::heds::SurfaceMesh& mesh;
	double ael;
public:
	Eigen::VectorXcd intrinData;
	Eigen::MatrixX3d extrinData;

	struct ArrowData {
		std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> pos;
		std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> color;
	} arrows;

	VectorField(kika::heds::SurfaceMesh& mesh, const Form form = Vwise, const Space space = Intrinsic) : form(form), space(space), mesh(mesh), ael(mesh.avgEdgeLength()) {
		if (space == Intrinsic) {
			if (form == Vwise)      intrinData = Eigen::VectorXcd::Zero(mesh.getNumVerts());
			else if (form == Fwise) intrinData = Eigen::VectorXcd::Zero(mesh.getNumFaces());
			else if (form == Ewise) intrinData = Eigen::VectorXcd::Zero(mesh.getNumEdges());
		} else {
			if (form == Vwise)      extrinData = Eigen::MatrixX3d::Zero(mesh.getNumVerts(), 3);
			else if (form == Fwise) extrinData = Eigen::MatrixX3d::Zero(mesh.getNumFaces(), 3);
			else if (form == Ewise) extrinData = Eigen::MatrixX3d::Zero(mesh.getNumEdges(), 3);
		}
	}

	VectorField(kika::heds::SurfaceMesh& mesh, const Eigen::MatrixX3d& data) : mesh(mesh), ael(mesh.avgEdgeLength()) {
		if (mesh.getNumVerts() == (size_t)data.rows()) {
			form = Vwise;
		} else if (mesh.getNumFaces() == (size_t)data.rows()) {
			form = Fwise;
		} else if (mesh.getNumFaces() == (size_t)data.rows()) {
			form = Ewise;
		} else {
			assert("VectorField : mesh element size != data.rows()");
		}

		space = Extrinsic;
		extrinData = data;
	}

	VectorField(kika::heds::SurfaceMesh& mesh, const Eigen::VectorXcd& data) : mesh(mesh), ael(mesh.avgEdgeLength()) {
		if (mesh.getNumVerts() == (size_t)data.rows()) {
			form = Vwise;
		} else if (mesh.getNumFaces() == (size_t)data.rows()) {
			form = Fwise;
		} else if (mesh.getNumFaces() == (size_t)data.rows()) {
			form = Ewise;
		} else {
			assert("VectorField : mesh element size != data.rows()");
		}

		space = Intrinsic;
		intrinData = data;
	}

	VectorField vwise() const {
		if (form == Vwise) {
			return *this;
		}

		size_t nv = mesh.getNumVerts();

		if (space == Intrinsic) {
			return extrin().vwise(); // (Face, Intrinsic) -> extrinsic ->vwise
		} else {
			Eigen::MatrixX3d vw = Eigen::MatrixX3d::Zero(nv, 3);

			if (form == Fwise) {
				for (size_t i = 0; i < nv; i++) {
					auto p = mesh.pos.row(i).transpose().eval();
					auto n = mesh.normal(mesh.verts[i]);
					double l = 0.0;
					for (const auto& f : mesh.verts[i]->adjacentFaces()) {
						double d = (mesh.center(mesh.faces[f]) - p).squaredNorm();
						l += d;
						vw.row(i).array() += d * util::project2plane(extrinData.row(f), n).array();
					}
					vw.row(i).array() /= l;
				}
			} else {

			}

			return VectorField(mesh, vw);
		}
	}

	VectorField fwise() const {
		if (form == Fwise) {
			return *this;
		}

		size_t nf = mesh.getNumFaces();

		if (space == Intrinsic) {
			return extrin().fwise(); // (Vwise, Intrinsic) -> extrin -> fwise

		} else {
			Eigen::MatrixX3d fw = Eigen::MatrixX3d::Zero(nf, 3);

			if (form == Vwise) {
				for (size_t i = 0; i < nf; i++) {
					Eigen::Vector3d n = mesh.normal(mesh.faces[i]);
					auto c = mesh.center(mesh.faces[i]);
					double l = 0.0;
					for (const auto& v : mesh.faces[i]->circuitVerts()) {
						double d = (mesh.pos.row(v).transpose() - c).squaredNorm();
						l += d;
						fw.row(i).array() += d * util::project2plane(extrinData.row(v), n).array();
					}
					fw.row(i).array() /= l;
				}
			} else {

			}

			return VectorField(mesh, fw);
		}

	}

	VectorField intrin() const {
		if (space == Intrinsic) {
			return *this;
		}

		size_t s = size();
		Eigen::VectorXcd field(s);

		if (form == Vwise) {
			for (size_t i = 0; i < s; i++) {
				Eigen::Vector3d ext = extrinData.row(i);

				auto n = mesh.normal(mesh.verts[i]);

				auto iedges = mesh.verts[i]->incidentEdges<heds::Ptr>();
				std::reverse(iedges.begin()+1, iedges.end());

				// find closest angle between ext and onerings
				double closestAngle = 1e+6;
				size_t closestIndex = 0;
				for (size_t j = 0; j < iedges.size(); j++) {
					auto edge = mesh.edgevec(iedges[j]->pair);
					auto angle = kika::util::angle(edge, ext);
					if (closestAngle > angle) {
						closestAngle = angle;
						closestIndex = j;
					}
				}

				// rotate from closest one to ext
				Eigen::Vector3d ref = mesh.edgevec(iedges[0]->pair);
				double oneringAngle = 0.0;
				for (size_t j = 1; j <= closestIndex; j++) {
					auto edge = mesh.edgevec(iedges[j]->pair);
					oneringAngle += kika::util::angle(ref, edge);
					ref = edge;
				}
				oneringAngle += kika::util::angle(ref, ext, n);

				field[i] = std::polar(ext.norm(), oneringAngle);
			}

		} else if (form == Fwise) {
			for (size_t i = 0; i < s; i++) {
				Eigen::Vector3d ext = extrinData.row(i);

				auto n = mesh.normal(mesh.faces[i]);
				auto c = mesh.center(mesh.faces[i]);

				auto averts = mesh.faces[i]->circuitVerts();

				double closestAngle = 1e+6;
				size_t closestIndex = 0;
				for (size_t j = 0; j < averts.size(); j++) {
					Eigen::Vector3d edge = (mesh.pos.row(averts[j]).transpose() - c);
					auto angle = kika::util::angle(edge, ext);
					if (closestAngle > angle) {
						closestAngle = angle;
						closestIndex = j;
					}
				}

				Eigen::Vector3d ref = (mesh.pos.row(averts[0]).transpose() - c);
				double oneringAngle = 0.0;
				for (size_t j = 1; j <= closestIndex; j++) {
					Eigen::Vector3d edge = (mesh.pos.row(averts[j]).transpose() - c);
					oneringAngle += kika::util::angle(ref, edge);
					ref = edge;
				}
				oneringAngle += kika::util::angle(ref, ext, n.normalized());

				field[i] = std::polar(ext.norm(), oneringAngle);
			}

		} else {

		}

		return VectorField(mesh, field);
	}

	VectorField extrin() const {
		if (space == Extrinsic) {
			return *this;
		}

		size_t s = size();
		Eigen::MatrixX3d field(s, 3);

		if (form == Vwise) {
			for (size_t i = 0; i < s; i++) {
				auto n = mesh.normal(mesh.verts[i]);
				auto e = kika::util::project2plane(mesh.edgevec(mesh.verts[i]->edge->pair).normalized(), n);
				field.row(i) = std::abs(intrinData[i]) * kika::util::quatRot(e, n, std::arg(intrinData[i]));
			}
		} else if (form == Fwise) {
			for (size_t i = 0; i < s; i++) {
				auto n = mesh.normal(mesh.faces[i]);
				auto c = mesh.center(mesh.faces[i]);
				Eigen::Vector3d e = (mesh.pos.row(mesh.faces[i]->edge->vert->id).transpose() - c).normalized();
				field.row(i) = std::abs(intrinData[i]) * kika::util::quatRot(e, n.normalized(), std::arg(intrinData[i]));
			}
		} else {

		}

		return VectorField(mesh, field);
	}

	size_t size() const {
		if (space == Intrinsic)
			return static_cast<size_t>(intrinData.rows());
		else
			return static_cast<size_t>(extrinData.rows());
	}

	Eigen::Vector3d operator()(int i) {
		assert("VectorField(i)" && i >= 0 && i < extrinData.rows());
		assert("VectorField form != Extrinsic" && space == Extrinsic);
		return extrinData.row(i);
	}

	Eigen::dcomplex operator[](int i) {
		assert("VectorField[i]" && i >= 0 && i < intrinData.rows());
		assert("VectorField form != Intrinsc" && space == Intrinsic);
		return intrinData[i];
	}

	/// @brief 
	/// @param offset : offset of normal direction rate (offset * ael)
	/// @param lengthRate : length rate (length = lengthRate * ael * (vf_length[0, 1] + baseLength))
	/// @param baseLength : length base (length = lengthRate * ael * (vf_length[0, 1] + baseLength))
	/// @param uniformLength
	/// @param rate (0, 1): distribution rate
	/// @param lineStyle
	/// @return 
	VectorField& prepareDrawableArrow(double offset = 0.1, double lengthRate = 1.0,
		double baseLength = 1.0, const bool uniformLength = true, const double rate = 1.0, const bool lineStyle = false) {
		offset *= ael;
		lengthRate *= ael;

		size_t s = size();
		arrows.pos.reserve(s);

		Eigen::VectorXd lengths = Eigen::VectorXd::Ones(s);

		std::random_device seed;
		std::mt19937 engine(seed());
		std::uniform_real_distribution<double> distrib(0.0, 1.0);

		auto append = [this, &lengthRate, &offset, &lineStyle, &distrib, &engine, &rate]
		(const double ls, Eigen::Vector3d& p, const Eigen::Vector3d& n, const Eigen::Vector3d& e) {
			if (distrib(engine) >= rate) return;

			p += n * offset;
			arrows.pos.emplace_back(p, p + e * lengthRate * ls);

			if (!lineStyle) {
				p += e * lengthRate * ls;
				Eigen::Vector3d r = kika::util::quatRot(e, n, std::numbers::pi/6.0*5.);
				arrows.pos.emplace_back(p + r * lengthRate * ls * 0.3, p);
				r = kika::util::quatRot(e, n, -std::numbers::pi/6.0*5.0);
				arrows.pos.emplace_back(p + r * lengthRate * ls * 0.3, p);
			}
		};

		if (space == Intrinsic) {
			lengths = (!uniformLength) * kika::eig::zero2oneNorm(intrinData.cwiseAbs()).array() + baseLength;

			if (form == Vwise) {
				for (size_t i = 0; i < s; i++) {
					auto p = mesh.pos.row(i).transpose().eval();
					auto n = mesh.normal(mesh.verts[i]);
					Eigen::Vector3d e = mesh.edgevec(mesh.verts[i]->edge->pair).normalized();
					e = kika::util::quatRot(e, n, std::arg(intrinData(i)));
					append(lengths[i], p, n, e);
				}
			} else if (form == Fwise) {
				for (size_t i = 0; i < s; i++) {
					auto p = mesh.center(mesh.faces[i]);
					Eigen::Vector3d n = mesh.normal(mesh.faces[i]).normalized();
					Eigen::Vector3d e = (mesh.pos.row(mesh.faces[i]->edge->vert->id).transpose() - p).normalized();
					e = kika::util::quatRot(e, n, std::arg(intrinData(i)));
					append(lengths[i], p, n, e);
				}
			} else {}
		} else {
			lengths = (!uniformLength) * kika::eig::zero2oneNorm(extrinData.rowwise().norm()).array() + baseLength;

			if (form == Vwise) {
				for (size_t i = 0; i < s; i++) {
					auto p = mesh.pos.row(i).transpose().eval();
					auto n = mesh.normal(mesh.verts[i]);
					Eigen::Vector3d e = extrinData.row(i).normalized();
					append(lengths[i], p, n, e);
				}
			} else if (form == Fwise) {
				for (size_t i = 0; i < s; i++) {
					auto p = mesh.center(mesh.faces[i]);
					Eigen::Vector3d n = mesh.normal(mesh.faces[i]).normalized();
					Eigen::Vector3d e = extrinData.row(i).normalized();
					append(lengths[i], p, n, e);
				}
			} else {}
		}

		arrows.color.resize(arrows.pos.size(), std::make_pair(Eigen::Vector3d(0.2, 0.0, 0.7), Eigen::Vector3d(0.7, 0.0, 0.2)));
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& os, const VectorField& vf);
};

std::ostream& operator<<(std::ostream& os, const VectorField& vf) {
	std::string form = "Vertex-wise";
	if (vf.form == Fwise) form = "Face-wise";
	else if (vf.form == Ewise) form = "Edge-wise";

	std::string space = "Intrinsic";
	if (vf.space == Extrinsic) space = "Extrinsic";

	std::cout << form << " " << space << " " << vf.size() << std::endl;
}
}