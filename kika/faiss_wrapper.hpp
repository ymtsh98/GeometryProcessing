#pragma once
#pragma warning (push) 
#pragma warning (disable : 4244)
#pragma warning (disable : 4251)
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/AutoTune.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>
#pragma warning(pop)
#include <Eigen/Core>

namespace kika::knn {
template <bool TRAIN = false>
class FaissKNN {
	const size_t dim;

	faiss::IndexFlatL2 quantizer;
	faiss::IndexIVFFlat index;

public:
	template <typename Derived, int Rows = Derived::RowsAtCompileTime, int Cols = Derived::ColsAtCompileTime>
	FaissKNN(const Eigen::DenseBase<Derived>& pts, const size_t ncells = 100, const size_t nprobe = 1) :
		dim(pts.cols()), quantizer(dim), index(&quantizer, dim, ncells) {

		Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor> tmp(pts.cast<float>());

		if constexpr (TRAIN) {
			index.train(tmp.rows(), tmp.data());
			index.add(tmp.rows(), tmp.data());
			index.nprobe = nprobe;
		} else {
			quantizer.add(tmp.rows(), tmp.data());
		}
	}

	void setNprobe(const size_t nprobe) {
		index.nprobe = nprobe;
	}

	template <typename T = size_t, typename Derived, int Rows = Derived::RowsAtCompileTime, int Cols = Derived::ColsAtCompileTime>
	void getKNearestNeighbors(const Eigen::DenseBase<Derived>& pts, const size_t k,
		std::vector<std::vector<T>>& indices, std::vector<std::vector<float>>& dists) {
		size_t n = (size_t)pts.rows();
		Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor> tmp(pts.cast<float>());
		std::vector<faiss::Index::idx_t> indices_(k * n);
		std::vector<float> dists_(k * n);

		if constexpr (TRAIN) index.search((faiss::Index::idx_t)n, tmp.data(), k, dists_.data(), indices_.data());
		else quantizer.search((faiss::Index::idx_t)n, tmp.data(), k, dists_.data(), indices_.data());

		indices.resize(n, std::vector<T>(k));
		dists.resize(n, std::vector<float>(k));
		for (size_t i = 0; i < n; i++) {
			for (size_t j = 0; j < k; j++) {
				indices[i][j] = static_cast<T>(indices_[i * k + j]);
				dists[i][j] = dists_[i * k + j];
			}
		}
	}

	template <typename T = size_t, typename Derived, int Rows = Derived::RowsAtCompileTime, int Cols = Derived::ColsAtCompileTime>
	std::vector<std::vector<T>> getRNearestNeighbors(const Eigen::DenseBase<Derived>& pts, const float r) {
		int n = (int)pts.rows();
		Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor> tmp(pts.cast<float>());

		faiss::RangeSearchResult* result = new faiss::RangeSearchResult(n);
		if constexpr (TRAIN) index.range_search(n, tmp.data(), r, result);
		else quantizer.range_search(n, tmp.data(), r, result);

		std::vector<std::vector<T>> indices(n);
		for (int i = 0; i < n; i++) {
			auto l = result->lims[i], r = result->lims[i + 1];
			indices[i].resize(r - l);
			for (size_t j = l, k = 0; j < r; j++, k++) {
				indices[i][k] = static_cast<T>(result->labels[j]);
			}
		}

		//std::cout <<  result->buffer_size << std::endl;
		//std::cout << result->lims[0] << " " << result->lims[1] << std::endl;
		//for (int i = 0; i < n; i++) {
		//	std::cout << "query " << i << " results:" << std::endl;
		//	for (size_t j = result->lims[i]; j < result->lims[i + 1]; j++) {
		//		std::cout << " " << result->labels[j] << ": " << result->distances[j] << std::endl;
		//	}
		//}

		delete result;
		return indices;
	}

	template <typename Derived, int Rows = Derived::RowsAtCompileTime, int Cols = Derived::ColsAtCompileTime>
	void getNearestNeighbors(const Eigen::DenseBase<Derived>& pts, std::vector<faiss::Index::idx_t>& indices, std::vector<float>& dists) {
		auto n = pts.rows();
		Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor> tmp(pts.cast<float>());
		indices.resize(n);
		dists.resize(n);

		if constexpr (TRAIN) index.search(n, tmp.data(), 1, dists.data(), indices.data());
		else quantizer.search(n, tmp.data(), 1, dists.data(), indices.data());
	}

	template <typename T = faiss::Index::idx_t, typename Derived>
	std::vector<T> getNearestNeighbors(const Eigen::DenseBase<Derived>& pts) {
		std::vector<faiss::Index::idx_t> indices;
		std::vector<float> dists;

		getNearestNeighbors(pts, indices, dists);

		if constexpr (std::is_same_v<T, faiss::Index::idx_t>)
			return indices;
		else
			return util::vecast<T>(indices);
	}

	template <typename T = faiss::Index::idx_t, typename Derived1, typename Derived2>
	static std::vector<T> nearestNeighbors(
		const Eigen::DenseBase<Derived1>& src,
		const Eigen::DenseBase<Derived2>& trg,
		const size_t ncells = 100, const size_t nprobe = 1) {
		return FaissKNN<TRAIN>(src, ncells, nprobe).getNearestNeighbors<T>(trg);
	}

};
}