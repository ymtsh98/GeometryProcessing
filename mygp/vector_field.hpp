#pragma once

namespace gp::vf {

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
	std::reference_wrapper<gp::heds::Mesh<Eigen::MatrixX3d>> mesh;

public:
	Eigen::VectorXcd intrinData;
	Eigen::MatrixX3d extrinData;

	VectorField(gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Form form = Vwise, const Space space = Intrinsic) : form(form), space(space), mesh(mesh) {
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

	VectorField(gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::MatrixX3d& data) : mesh(mesh){
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

	VectorField(gp::heds::Mesh<Eigen::MatrixX3d>& mesh, const Eigen::VectorXcd& data) : mesh(mesh) {
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

		size_t nv = mesh.get().getNumVerts();

		if (space == Intrinsic) {
			return extrin().vwise();

		} else {
			Eigen::MatrixX3d vw = Eigen::MatrixX3d::Zero(nv, 3);

			if (form == Fwise) {
				for (size_t i = 0; i < nv; i++) {
					auto p = mesh.get().position(i);
					auto n = mesh.get().normal(mesh.get().verts[i]);
					double l = 0.0;
					for (const auto& f : mesh.get().verts[i]->adjacentFindex()) {
						double d = (mesh.get().center(mesh.get().faces[f]) - p).squaredNorm();
						l += d;
						vw.row(i).array() += d * util::project(extrinData.row(f), n).array();
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

		size_t nf = mesh.get().getNumFaces();

		if (space == Intrinsic) {
			return extrin().fwise();

		} else {
			Eigen::MatrixX3d fw = Eigen::MatrixX3d::Zero(nf, 3);

			if (form == Vwise) {
				for (size_t i = 0; i < nf; i++) {
					Eigen::Vector3d n = mesh.get().normal(mesh.get().faces[i]);
					auto c = mesh.get().center(mesh.get().faces[i]);
					double l = 0.0;
					for (const auto& v : mesh.get().faces[i]->adjacentVindex()) {
						double d = (mesh.get().position(v) - c).squaredNorm();
						l += d;
						fw.row(i).array() += d * util::project(extrinData.row(v), n).array();
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
		Eigen::VectorXcd field(s, 3);

		if (form == Vwise) {
			for (size_t i = 0; i < s; i++) {
				Eigen::Vector3d ext = extrinData.row(i);

				auto n = mesh.get().normal(mesh.get().verts[i]);

				auto iedges = mesh.get().verts[i]->incidentEdges();
				std::reverse(iedges.begin()+1, iedges.end());

				double closestAngle = 1e+6;
				size_t closestIndex = 0;
				for (size_t j = 0; j < iedges.size(); j++) {
					auto edge = mesh.get().edge2vec(iedges[j]->pair);
					auto angle = gp::util::angle(edge, ext);
					if (closestAngle > angle) {
						closestAngle = angle;
						closestIndex = j;
					}
				}

				Eigen::Vector3d ref = mesh.get().edge2vec(iedges[0]->pair);
				double oneringAngle = 0.0;
				for (size_t j = 1; j <= closestIndex; j++) {
					auto edge = mesh.get().edge2vec(iedges[j]->pair);
					oneringAngle += gp::util::angle(ref, edge);
					ref = edge;
				}
				oneringAngle += gp::util::angle(ref, ext, n);

				field[i] = std::polar(ext.norm(), oneringAngle);
			}

		} else if (form == Fwise) {
			for (size_t i = 0; i < s; i++) {
				Eigen::Vector3d ext = extrinData.row(i);

				auto n = mesh.get().normal(mesh.get().faces[i]);
				auto c = mesh.get().center(mesh.get().faces[i]);

				auto averts = mesh.get().faces[i]->adjacentVindex();

				double closestAngle = 1e+6;
				size_t closestIndex = 0;
				for (size_t j = 0; j < averts.size(); j++) {
					Eigen::Vector3d edge = (mesh.get().position(averts[j]) - c);
					auto angle = gp::util::angle(edge, ext);
					if (closestAngle > angle) {
						closestAngle = angle;
						closestIndex = j;
					}
				}

				Eigen::Vector3d ref = (mesh.get().position(averts[0]) - c);
				double oneringAngle = 0.0;
				for (size_t j = 1; j <= closestIndex; j++) {
					Eigen::Vector3d edge = (mesh.get().position(averts[j]) - c);
					oneringAngle += gp::util::angle(ref, edge);
					ref = edge;
				}
				oneringAngle += gp::util::angle(ref, ext, n.normalized());

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
				auto n = mesh.get().normal(mesh.get().verts[i]);
				auto e = gp::util::project(mesh.get().edge2vec(mesh.get().verts[i]->edge->pair).normalized(), n);
				field.row(i) = std::abs(intrinData[i]) * gp::util::quatRot(e, n, std::arg(intrinData[i]));
			}
		} else if (form == Fwise) {
			for (size_t i = 0; i < s; i++) {
				auto n = mesh.get().normal(mesh.get().faces[i]);
				auto c = mesh.get().center(mesh.get().faces[i]);
				Eigen::Vector3d e = (mesh.get().position(mesh.get().faces[i]->edge->vert->id) - c).normalized();
				field.row(i) = std::abs(intrinData[i]) * gp::util::quatRot(e, n.normalized(), std::arg(intrinData[i]));
			}
		} else {

		}

		return VectorField(mesh, field);
	}

	/*
	VectorField& convert(const Form form) {
		if (this->form == form) return *this;

		if (form == Vwise) {
			this->form = Vwise;

		} else if (form == Fwise) {
			this->form = Fwise;

		} else if (form == Ewise) {
			this->form = Ewise;

		}

		return *this;
	}

	VectorField& convert(const Space space) {
		if (this->space == space) return *this;
		
		if (space == Intrinsic) {
			this->space = Intrinsic;
		} else {
			this->space = Extrinsic;
		}

		return *this;
	}

	VectorField& convert(const Form form, const Space space) {
		return convert(form).convert(space);
	}
	*/

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

	std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> drawableArrow(const double offset = 1.0, const double baseLength = 1.0, 
		const double uniformLength = 1.0, const bool lineStyle = false, const double rate = 1.0) {
		size_t s = size();
		std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> arrows;
		arrows.reserve(s);
		
		Eigen::VectorXd lengths = Eigen::VectorXd::Ones(s);

		std::random_device seed;
		std::mt19937 engine(seed());
		std::uniform_real_distribution<double> distrib(0.0, 1.0);

		auto append = [&arrows, &baseLength, &offset, &lineStyle, &distrib, &engine, & rate]
		(const double ls, Eigen::Vector3d& p, const Eigen::Vector3d& n, const Eigen::Vector3d& e) {
			if (distrib(engine) >= rate) return;

			p += n * offset;
			arrows.emplace_back(p, p + e * baseLength * ls);

			if (!lineStyle) {
				p += e * baseLength * ls;
				Eigen::Vector3d r = gp::util::quatRot(e, n, std::numbers::pi/6.0*5.);
				arrows.emplace_back(p + r * baseLength * ls * 0.3, p);
				r = gp::util::quatRot(e, n, -std::numbers::pi/6.0*5.0);
				arrows.emplace_back(p + r * baseLength * ls * 0.3, p);
			}
		};

		if (space == Intrinsic) {
			if (uniformLength < 1.0) {
				lengths = gp::eig::zeroOneNorm(intrinData.cwiseAbs()).array() + uniformLength;
			}

			if (form == Vwise) {
				for (size_t i = 0; i < s; i++) {
					auto p = mesh.get().position(i);
					auto n = mesh.get().normal(mesh.get().verts[i]);
					auto e = mesh.get().edge2vec(mesh.get().verts[i]->edge->pair);
					e = gp::util::quatRot(e.normalized(), n, std::arg(intrinData(i)));
					append(lengths[i], p, n, e);
				}
			} else if (form == Fwise) {
				for (size_t i = 0; i < s; i++) {
					auto p = mesh.get().center(mesh.get().faces[i]);
					Eigen::Vector3d n = mesh.get().normal(mesh.get().faces[i]).normalized();
					Eigen::Vector3d e = (mesh.get().position(mesh.get().faces[i]->edge->vert->id) - p).normalized();
					e = gp::util::quatRot(e, n, std::arg(intrinData(i)));
					append(lengths[i], p, n, e);
				}
			} else {}
		} else {
			if (uniformLength < 1.0) {
				lengths = gp::eig::zeroOneNorm(extrinData.rowwise().norm()).array() + uniformLength;
			}

			if (form == Vwise) {
				for (size_t i = 0; i < s; i++) {
					auto p = mesh.get().position(i);
					auto n = mesh.get().normal(mesh.get().verts[i]);
					Eigen::Vector3d e = extrinData.row(i).normalized();
					append(lengths[i], p, n, e);
				}
			} else if (form == Fwise) {
				for (size_t i = 0; i < s; i++) {
					auto p = mesh.get().center(mesh.get().faces[i]);
					Eigen::Vector3d n = mesh.get().normal(mesh.get().faces[i]).normalized();
					Eigen::Vector3d e = extrinData.row(i).normalized();
					append(lengths[i], p, n, e);
				}
			} else {}
		}

		return arrows;
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