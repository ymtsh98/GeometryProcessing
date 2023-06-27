#pragma once
#include <nanoflann.hpp>

namespace kika::knn {

/// nanoflann wrapper
/// warning : this class duplicate original point sets.
template <typename Derived, typename Scalar = Derived::Scalar,
	int Rows = Derived::RowsAtCompileTime, int Cols = Derived::ColsAtCompileTime>
	class NanoFlannKNN {
	//typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatXf;
	//nanoflann::KDTreeEigenMatrixAdaptor<RowMatXf> tree;

	typedef Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor> RowMatXf;
	RowMatXf coords;
	nanoflann::KDTreeEigenMatrixAdaptor<RowMatXf> tree;

	const int maxLeaf;

	public:

		NanoFlannKNN(const Eigen::DenseBase<Derived>& pts, const int maxLeaf = 10) : coords(pts.cast<float>()), tree(pts.cols(), coords, maxLeaf), maxLeaf(maxLeaf) {}

		void getNearestNeighbor(const float* query, size_t& index, float& dist) const {
			nanoflann::KNNResultSet<float> resultSet(1);
			resultSet.init(&index, &dist);
			tree.index->findNeighbors(resultSet, query, nanoflann::SearchParameters());
		}

		size_t getNearestNeighbor(const float* query) const {
			size_t index;
			float dist;
			getNearestNeighbor(query, index, dist);

			return index;
		}

		template <typename T = size_t, typename Deriv, int R = Deriv::RowsAtCompileTime, int C = Deriv::ColsAtCompileTime>
		std::vector<T> getNearestNeighbors(const Eigen::DenseBase<Deriv>& pts) const {
			std::vector<T> nn(pts.rows());

			Eigen::Matrix<float, R, C, Eigen::RowMajor> tmp = pts.cast<float>();

			#pragma omp parallel for
			for (int i = 0; i < tmp.rows(); ++i) {
				nn[i] = static_cast<T>(getNearestNeighbor(tmp.row(i).data()));
			}

			return nn;
		}

		void getKNearestNeighbor(const float* query, const size_t k,
			std::vector<size_t>& indices, std::vector<float>& dists) const {
			indices.resize(k);
			dists.resize(k);

			nanoflann::KNNResultSet<float> resultSet(k);
			resultSet.init(&indices[0], &dists[0]);
			tree.index->findNeighbors(resultSet, query, nanoflann::SearchParameters());
		}

		template <typename T = size_t>
		std::vector<T> getKNearestNeighbor(const float* query, const size_t k) const {
			std::vector<size_t> indices(k);
			std::vector<float> dists(k);
			getKNearestNeighbor(query, k, indices, dists);
			if constexpr (std::is_same_v<T, size_t>)
				return indices;
			else
				return util::vecast<T>(indices);
		}

		template <typename Deriv, int R = Deriv::RowsAtCompileTime, int C = Deriv::ColsAtCompileTime>
		void getKNearestNeighbors(const Eigen::DenseBase<Deriv>& pts, const size_t k,
			std::vector<std::vector<size_t>>& indices, std::vector<std::vector<float>>& dists) {
			indices.resize(pts.rows(), std::vector<size_t>(k));
			dists.resize(pts.rows(), std::vector<float>(k));

			Eigen::Matrix<float, R, C, Eigen::RowMajor> tmp = pts.cast<float>();

			#pragma omp parallel for
			for (int i = 0; i < tmp.rows(); ++i) {
				getKNearestNeighbor(tmp.row(i).data(), k, indices[i], dists[i]);
			}
		}

		void getRNearestNeighbor(const float* query, const float r,
			std::vector<std::pair<Eigen::Index, float>>& indicesDists) const {
			tree.index->radiusSearch(query, r, indicesDists, nanoflann::SearchParameters());
		}

		template <typename Deriv, int R = Deriv::RowsAtCompileTime, int C = Deriv::ColsAtCompileTime>
		void getRNearestNeighbors(const Eigen::DenseBase<Deriv>& pts, const float r,
			std::vector<std::vector<std::pair<Eigen::Index, float>>>& indicesDists) const {
			indicesDists.resize(pts.rows());

			Eigen::Matrix<float, R, C, Eigen::RowMajor> tmp = pts.cast<float>();

			#pragma omp parallel for
			for (int i = 0; i < tmp.rows(); ++i) {
				getRNearestNeighbor(tmp.row(i).data(), r, indicesDists[i]);
			}
		}

		template <typename T = size_t, typename Deriv, int R = Deriv::RowsAtCompileTime, int C = Deriv::ColsAtCompileTime>
		std::vector<std::vector<T>> getRNearestNeighbors(const Eigen::DenseBase<Deriv>& pts, const float r) const {
			std::vector<std::vector<std::pair<Eigen::Index, float>>> indicesDists;
			getRNearestNeighbors(pts, r, indicesDists);

			std::vector<std::vector<T>> indices(indicesDists.size());
			for (size_t i = 0; i < indices.size(); i++) {
				indices[i].resize(indicesDists[i].size());
				for (size_t j = 0; j < indices[i].size(); j++) {
					indices[i][j] = static_cast<T>(indicesDists[i][j].first);
				}
			}
			return indices;
		}

};

template <typename T = size_t, typename Derived1, typename Derived2>
std::vector<T> nanoflann(const Eigen::DenseBase<Derived1>& src, const Eigen::DenseBase<Derived2>& trg, const int maxLeaf = 10) {
	return NanoFlannKNN(src, maxLeaf).getNearestNeighbors<T>(trg);
}

}