#pragma once
#include <nanoflann.hpp>
namespace gp::spart {
/*
nanoflann wrapper
warning : this class duplicate original point sets.
*/
class NanoFlannKNN {
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatXf;
	RowMatXf coords;
	nanoflann::KDTreeEigenMatrixAdaptor<RowMatXf> tree;

	const int maxLeaf;

public:
	NanoFlannKNN(const Eigen::MatrixXf& pts, const int maxLeaf = 10) : coords(pts.leftCols(pts.cols())),
		tree(pts.cols(), coords, maxLeaf), maxLeaf(maxLeaf) {}

	inline void getNearestNeighbor(const Eigen::VectorXf& query, size_t& index, float& dist) const {
		nanoflann::KNNResultSet<float> resultSet(1);
		resultSet.init(&index, &dist);
		tree.index->findNeighbors(resultSet, query.data(), nanoflann::SearchParams());
	}

	inline size_t getNearestNeighbor(const Eigen::VectorXf& query) const {
		size_t index;
		float dist;
		getNearestNeighbor(query, index, dist);

		return index;
	}

	std::vector<size_t> getNearestNeighbors(const Eigen::MatrixXf& pts) const {
		std::vector<size_t> nn(pts.rows());

		#pragma omp parallel for
		for (int i = 0; i < pts.rows(); ++i) {
			nn[i] = getNearestNeighbor(pts.row(i));
		}

		return nn;
	}

	inline void getKNearestNeighbor(const Eigen::VectorXf& query, const size_t k,
		std::vector<size_t>& indices, std::vector<float>& dists) const {
		indices.resize(k);
		dists.resize(k);

		nanoflann::KNNResultSet<float> resultSet(k);
		resultSet.init(&indices[0], &dists[0]);
		tree.index->findNeighbors(resultSet, query.data(), nanoflann::SearchParams());
	}

	inline std::vector<size_t> getKNearestNeighbor(const Eigen::VectorXf& query, const size_t k) const {
		std::vector<size_t> indices(k);
		std::vector<float> dists(k);
		getKNearestNeighbor(query, k, indices, dists);
		return indices;
	}

	void getKNearestNeighbors(const Eigen::MatrixXf& pts, const size_t k,
		std::vector<std::vector<size_t>>& indices, std::vector<std::vector<float>>& dists) {
		indices.resize(pts.rows(), std::vector<size_t>(k));
		dists.resize(pts.rows(), std::vector<float>(k));

		#pragma omp parallel for
		for (int i = 0; i < pts.rows(); ++i) {
			getKNearestNeighbor(pts.row(i), k, indices[i], dists[i]);
		}
	}

	inline void getRNearestNeighbor(const Eigen::VectorXf& query, const float r,
		std::vector<std::pair<Eigen::Index, float>>& indicesDists) const {
		tree.index->radiusSearch(query.data(), r, indicesDists, nanoflann::SearchParams());
	}

	void getRNearestNeighbors(const Eigen::MatrixXf& pts, const float r,
		std::vector<std::vector<std::pair<Eigen::Index, float>>>& indicesDists) const {
		indicesDists.resize(pts.rows());

		#pragma omp parallel for
		for (int i = 0; i < pts.rows(); ++i) {
			getRNearestNeighbor(pts.row(i), r, indicesDists[i]);
		}
	}
};
}