#pragma once
#include "geoproc.hpp"
#include "halfedge.hpp"
#if __has_include("Eigen/Eigen")
#define EIGEN_INCLUDE
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/Sparse"
#endif

namespace gp {
	Eigen::VectorXd meanCurvature(const heds::Mesh& mesh, const Eigen::SparseMatrix<double>& lbo) {
		Eigen::VectorXd curvature = Eigen::VectorXd::Zero(mesh.getNumVertices());
		Eigen::MatrixXd verts = Eigen::MatrixXd(mesh.getNumVertices(), 3);

		for (size_t i = 0; i < mesh.getNumVertices(); i++) verts.row(i) = mesh.verts[i]->pos.eig;
		Eigen::MatrixXd H = lbo * verts;
		for (size_t i = 0; i < mesh.getNumVertices(); i++) curvature(i) = 0.5 * H.row(i).norm();

		return curvature;
	}

	Eigen::VectorXd gaussianCurvature(const heds::Mesh& mesh) {
		Eigen::VectorXd curvature = Eigen::VectorXd::Zero(mesh.getNumVertices());
		curvature.fill(2.0 * std::numbers::pi);

		for (auto& face : mesh.faces) {
			auto v1 = face->edge->vert;
			auto v2 = face->edge->next->vert;
			auto v3 = face->edge->next->next->vert;

			auto l12 = v2->pos.eig - v1->pos.eig;
			auto l13 = v3->pos.eig - v1->pos.eig;
			auto l21 = v1->pos.eig - v2->pos.eig;
			auto l23 = v3->pos.eig - v2->pos.eig;
			auto l31 = v1->pos.eig - v3->pos.eig;
			auto l32 = v2->pos.eig - v3->pos.eig;

			auto theta1 = acos(l12.dot(l13) / (l12.norm() * l13.norm()));
			auto theta2 = acos(l21.dot(l23) / (l21.norm() * l23.norm()));
			auto theta3 = acos(l31.dot(l32) / (l31.norm() * l32.norm()));

			curvature[v1->id] -= theta1;
			curvature[v2->id] -= theta2;
			curvature[v3->id] -= theta3;
		}

		auto area = areaVectorPerVert(mesh);
		curvature.array() /= area.array();
		return curvature;
	}

	std::vector<double> hksParams(const double min_eig_val, const double max_eig_val, const double num_sample) {
		std::vector<double> params(num_sample);
		double min_param = abs(4.0 * log(10.0) / max_eig_val);
		double max_param = abs(4.0 * log(10.0) / min_eig_val);
		double step = log(max_param) - log(min_param);

		for (size_t i = 0; i < num_sample; i++) {
			params[i] = exp(log(min_param) + step * (i + 1) / (double)num_sample);
		}

		return params;
	}

	Eigen::VectorXd heatKernelSignature(const Eigen::VectorXd& eigenvals,
																			const Eigen::MatrixXd& eigenvecs,
																			const double time, const bool scaled = false, const int y = -1) {

		Eigen::VectorXd heat = Eigen::VectorXd::Zero(eigenvecs.rows());
		size_t numeigs = eigenvals.size();
		double scale = 0.0;

		if (y == -1) {
			for (size_t i = 1; i < numeigs; i++) {
				scale += exp(-eigenvals[i] * time);
				heat = heat.array() + exp(-eigenvals[i] * time) * Eigen::pow(eigenvecs.col(i).array(), 2);
			}
			if (scaled) heat.array() /= scale;

		} else {
			for (size_t i = 1; i < numeigs; i++) {
				for (size_t x = 0; x < eigenvecs.rows(); x++) {
					heat[x] += exp(-eigenvals[i] * time) * eigenvecs(x, i) * eigenvecs(y, i);
				}
			}

		}

		return heat;
	}

	std::pair<std::vector<double>, double> wksParams(const double min_eig_val, const double max_eig_val, const double num_sample) {
		std::vector<double> params(num_sample);
		double min_param = log(max_eig_val) / 1.02;
		double max_param = log(min_eig_val);
		double step = max_param - min_param;

		for (size_t i = 0; i < num_sample; i++) {
			params[i] = min_param + step * (i + 1) / (double)num_sample;
		}

		double sigma = num_sample > 0 ? (params[1] - params[0]) * 5.0 : 0.0;

		return { params, sigma };
	}

	Eigen::VectorXd waveKernelSignature(const Eigen::VectorXd& eigenvals,
																			const Eigen::MatrixXd& eigenvecs,
																			const double energy, const double sigma) {

		Eigen::VectorXd wave = Eigen::VectorXd::Zero(eigenvecs.rows());
		size_t numeigs = eigenvals.size();

		double scale = 0.0;
		for (size_t i = 1; i < numeigs; i++) {
			scale += exp(-pow(energy - log(eigenvals(i)), 2) / (2.0 * sigma * sigma));
			double c = exp(-pow(energy - log(eigenvals(i)), 2) / (2.0 * sigma * sigma));
			wave.array() += Eigen::pow(eigenvecs.col(i).array(), 2) * c;
		}

		wave.array() /= scale;

		return wave;
	}

	Eigen::MatrixXd timeStepHKS(const Eigen::VectorXd& eigenvals,
															const Eigen::MatrixXd& eigenvecs,
															const size_t num_sample) {
		Eigen::MatrixXd hks(eigenvecs.rows(), num_sample);

		size_t numeigs = eigenvals.size();
		auto energy = hksParams(eigenvals[1], eigenvals[numeigs - 1], num_sample);

		Eigen::MatrixXd phi2 = Eigen::pow(eigenvecs.block(0, 1, eigenvecs.rows(), numeigs - 1).array(), 2.0);
		Eigen::VectorXd E = eigenvals.block(1, 0, numeigs - 1, 1);
		for (size_t i = 0; i < num_sample; i++) {
			Eigen::VectorXd C = Eigen::exp(-energy[i] * E.array());

			hks.col(i).array() = (phi2.array().rowwise() * C.transpose().array()).rowwise().sum() / C.sum();
		}

		return hks;
	}

	Eigen::MatrixXd timeStepWKS(const Eigen::VectorXd& eigenvals,
															const Eigen::MatrixXd& eigenvecs,
															const size_t num_sample) {
		
		Eigen::MatrixXd wks(eigenvecs.rows(), num_sample);
		
		size_t numeigs = eigenvals.size();
		auto [energy, sigma] = wksParams(eigenvals[1], eigenvals[numeigs-1], num_sample);
		
		Eigen::MatrixXd phi2 = Eigen::pow(eigenvecs.block(0, 1, eigenvecs.rows(), numeigs-1).array(), 2.0);
		Eigen::VectorXd E = Eigen::log(eigenvals.array()).block(1, 0, numeigs-1, 1);

		for (size_t i = 0; i < num_sample; i++) {
			Eigen::VectorXd C = Eigen::exp(-Eigen::pow(energy[i] * Eigen::VectorXd::Ones(numeigs-1).array() 
																								- E.array(), 2.0) / (2.0 * sigma * sigma));
			
			wks.col(i).array() = (phi2.array().rowwise() * C.transpose().array()).rowwise().sum() / C.sum();
		}

		return wks;
	}

	Eigen::MatrixXd timeStepMC(const heds::Mesh& mesh, const Eigen::SparseMatrix<double>& lbo, const size_t num_sample, const double time = 1.0) {
		Eigen::MatrixXd mc(mesh.getNumVertices(), num_sample);
		auto curvature = meanCurvature(mesh, lbo);
		for (size_t i = 0; i < num_sample; i++) {
			mc.col(i) = curvature;
			curvature = heatDiffusionSmoothing(lbo, curvature, time);
		}
		return mc;
	}

	Eigen::MatrixXd timeStepGC(const heds::Mesh& mesh, const Eigen::SparseMatrix<double>& lbo, const size_t num_sample, const double time = 1.0) {
		Eigen::MatrixXd gc(mesh.getNumVertices(), num_sample);
		auto curvature = gaussianCurvature(mesh);
		for (size_t i = 0; i < num_sample; i++) {
			gc.col(i) = curvature;
			curvature = heatDiffusionSmoothing(lbo, curvature, time);
		}
		return gc;
	}
}