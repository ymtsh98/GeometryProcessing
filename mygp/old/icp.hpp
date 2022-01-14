#pragma once
#include <tuple>
#include <vector>

#if __has_include("glm/glm.hpp")
#include <glm/glm.hpp>
#define GLM_INCLUDE
#endif

#if __has_include("Eigen/Eigen")
#include "Eigen/Core"
#define EIGEN_INCLUDE
#endif

namespace gp {
	template <typename C>
	auto rigidTransform(const std::vector<glm::vec3>& src_pos, const std::vector<glm::vec3>& trg_pos, const C& correspondence) {
		glm::vec3 src_center{ 0.f, 0.f, 0.f }, trg_center{ 0.f, 0.f, 0.f };
		for (auto& v : src_pos) src_center += v;
		for (auto& v : trg_pos) trg_center += v;
		src_center /= (float)src_pos.size();
		trg_center /= (float)trg_pos.size();

		Eigen::MatrixXd X = Eigen::MatrixXd::Zero(3, trg_pos.size());
		Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(3, trg_pos.size());

		for (size_t i = 0; i < trg_pos.size(); i++) {
			auto v = src_pos[correspondence[i]] - src_center;
			X.col(i) = Eigen::Vector3d(v.x, v.y, v.z);
		}
		for (size_t i = 0; i < trg_pos.size(); i++) {
			auto v = trg_pos[i] - trg_center;
			Y.col(i) = Eigen::Vector3d(v.x, v.y, v.z);
		}

		auto S = X * Y.transpose();
		Eigen::JacobiSVD<Eigen::Matrix3d> SVD(S, Eigen::ComputeThinU | Eigen::ComputeFullV);
		auto V = SVD.matrixV();
		auto U = SVD.matrixU();

		Eigen::DiagonalMatrix<double, 3> D(Eigen::Vector3d{ 1., 1., (V * U.transpose()).determinant() });
		Eigen::MatrixXd R = V * D * U.transpose();
		Eigen::Vector3d T = Eigen::Vector3d{ trg_center[0], trg_center[1], trg_center[2] } -
			R * Eigen::Vector3d{ src_center[0], src_center[1], src_center[2] };
		
		glm::mat3 glmR{ R(0, 0), R(0, 1), R(0, 2),
										R(1, 0), R(1, 1), R(1, 2),
										R(2, 0), R(2, 1), R(2, 2)};
		glm::vec3 glmT = { T(0), T(1), T(2) };

		return std::pair(glmR, glmT);
	}

	#ifdef EIGEN_INCLUDE
	template <typename C>
	auto rigidTransform(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg, const C& correspondence, bool s2t = true) {
		Eigen::VectorXd src_center = src.rowwise().sum() / src.cols();
		Eigen::VectorXd trg_center = trg.rowwise().sum() / trg.cols();

		Eigen::MatrixXd X = Eigen::MatrixXd::Zero(trg.rows(), trg.cols());
		Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(trg.rows(), trg.cols());

		for (size_t i = 0; i < trg.cols(); i++) X.col(i) = src.col(correspondence[i]) - src_center;
		for (size_t i = 0; i < trg.cols(); i++) Y.col(i) = trg.col(i) - trg_center;
		
		auto S = s2t ? X * Y.transpose() : Y * X.transpose();
		Eigen::JacobiSVD<Eigen::MatrixXd> SVD(S, Eigen::ComputeThinU | Eigen::ComputeFullV);
		auto V = SVD.matrixV();
		auto U = SVD.matrixU();

		Eigen::VectorXd D = Eigen::VectorXd::Ones(trg.rows());
		D(trg.rows()-1) = (V * U.transpose()).determinant();

		Eigen::MatrixXd R = (V * D.asDiagonal() * U.transpose()).transpose();
		Eigen::VectorXd T = trg_center - R * src_center;

		return std::pair(R, T);
	}

	template <typename C>
	auto rigidTransformSVD(const Eigen::MatrixXd& src, const Eigen::MatrixXd& trg, const C& correspondence, bool s2t = true) {
		Eigen::MatrixXd X = Eigen::MatrixXd::Zero(trg.rows(), trg.cols());
		Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(trg.rows(), trg.cols());

		for (size_t i = 0; i < trg.cols(); i++) X.col(i) = src.col(correspondence[i]);
		for (size_t i = 0; i < trg.cols(); i++) Y.col(i) = trg.col(i);

		auto S = s2t ? X * Y.transpose() : Y * X.transpose();
		Eigen::JacobiSVD<Eigen::MatrixXd> SVD(S, Eigen::ComputeThinU | Eigen::ComputeFullV);
		auto V = SVD.matrixV();
		auto U = SVD.matrixU();

		return std::pair(U, V);
	}
	#endif
}