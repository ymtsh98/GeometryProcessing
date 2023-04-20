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

namespace gp::spart {
#ifndef MYGP_IMPLEMENTATION
template <bool TRAIN = false>
class FaissKNN {
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatXf;
	const RowMatXf coords;
	const size_t d;

	faiss::IndexFlatL2 quantizer;
	faiss::IndexIVFFlat index;

public:
	FaissKNN(const Eigen::MatrixXf& pts, const size_t ncells = 100, const size_t nprobe = 1);

	void setNprobe(const size_t nprobe);

	void getKNearestNeighbors(const Eigen::MatrixXf& pts, const size_t k,
		std::vector<std::vector<size_t>>& indices, std::vector<std::vector<float>>& dists);

	void getNearestNeighbors(const Eigen::MatrixXf& pts, std::vector<faiss::Index::idx_t>& indices, std::vector<float>& dists);

	std::vector<faiss::Index::idx_t> getNearestNeighbors(const Eigen::MatrixXf& pts);

	//TODO
	//void getRNearestNeighbors(const Eigen::MatrixXf& pts, const double r);

	static std::vector<faiss::Index::idx_t> nearestNeighbors(const Eigen::MatrixXf& src, const Eigen::MatrixXf& trg, const size_t ncells = 100, const size_t nprobe = 1);

};

#else
#define MYGP_IMPLEMENTATION

template <bool TRAIN = false>
class FaissKNN {
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatXf;
	const RowMatXf coords;
	const size_t d;

	faiss::IndexFlatL2 quantizer;
	faiss::IndexIVFFlat index;

public:
	FaissKNN(const Eigen::MatrixXf& pts, const size_t ncells = 100, const size_t nprobe = 1) :
		coords(pts.leftCols(pts.cols())), d(pts.cols()), quantizer(d), index(&quantizer, d, ncells) {

		if constexpr (TRAIN) {
			index.train(coords.rows(), coords.data());
			index.add(coords.rows(), coords.data());
			index.nprobe = nprobe;
		} else {
			quantizer.add(coords.rows(), coords.data());
		}
	}

	void setNprobe(const size_t nprobe) {
		index.nprobe = nprobe;
	}

	void getKNearestNeighbors(const Eigen::MatrixXf& pts, const size_t k,
														std::vector<std::vector<size_t>>& indices, std::vector<std::vector<float>>& dists) {
		int n = pts.rows();
		RowMatXf rowPts(pts.leftCols(pts.cols()));
		std::vector<faiss::Index::idx_t> indices_(k * n);
		std::vector<float> dists_(k * n);

		if constexpr (TRAIN) index.search(n, rowPts.data(), k, dists_.data(), indices_.data());
		else quantizer.search(n, rowPts.data(), k, dists_.data(), indices_.data());

		indices.resize(n, std::vector<size_t>(k));
		dists.resize(n, std::vector<float>(k));
		for (size_t i = 0; i < n; i++) {
			for (size_t j = 0; j < k; j++) {
				indices[i][j] = indices_[i * k + j];
				dists[i][j] = dists_[i * k + j];
			}
		}
	}

	void getNearestNeighbors(const Eigen::MatrixXf& pts, std::vector<faiss::Index::idx_t>& indices, std::vector<float>& dists) {
		int n = (int)pts.rows();
		RowMatXf rowPts(pts.leftCols(pts.cols()));
		indices.resize(n);
		dists.resize(n);

		if constexpr (TRAIN) index.search(n, rowPts.data(), 1, dists.data(), indices.data());
		else quantizer.search(n, rowPts.data(), 1, dists.data(), indices.data());
	}

	std::vector<faiss::Index::idx_t> getNearestNeighbors(const Eigen::MatrixXf& pts) {
		std::vector<faiss::Index::idx_t> indices;
		std::vector<float> dists;

		getNearestNeighbors(pts, indices, dists);

		return indices;
	}

	//TODO
	void getRNearestNeighbors(const Eigen::MatrixXf& pts, const double r) {
		//int n = pts.rows();
		//RowMatXf rowPts(pts.leftCols(pts.cols()));
		//faiss::RangeSearchResult* result = new faiss::RangeSearchResult(n);
		//if constexpr (TRAIN) index.range_search(n, rowPts.data(), r, result);
		//else quantizer.range_search(n, rowPts.data(), r, result);
		//std::cout <<  result->buffer_size << std::endl;
		//std::cout << result->lims[0] << " " << result->lims[1] << std::endl;
		//
		//for (int i = 0; i < n; i++) {
		//	std::cout << "query " << i << " results:" << std::endl;
		//	for (size_t j = result->lims[i]; j < result->lims[i + 1]; j++) {
		//		std::cout << result->labels[j] << ": " << result->distances[j] << "  ";
		//	}
		//}
		//
		//delete result;
	}

	static std::vector<faiss::Index::idx_t> nearestNeighbors(const Eigen::MatrixXf& src, const Eigen::MatrixXf& trg, const size_t ncells = 100, const size_t nprobe = 1) {
		auto tree = FaissKNN<TRAIN>(src, ncells, nprobe);
		return tree.getNearestNeighbors(trg);
	}
};

#endif
}