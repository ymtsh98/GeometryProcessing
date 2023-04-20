#pragma once
#if __has_include("small_eigs.h")
#pragma comment(lib, "mclmcrrt.lib")
#pragma comment(lib, "small_eigs.lib")
#include <small_eigs.h>

namespace kika::mlab {
inline void initMatlabEigs() {
	static bool matlabEigsFirstInitialization = false;
	if (!matlabEigsFirstInitialization) matlabEigsFirstInitialization = small_eigsInitialize();
}

inline std::pair<Eigen::VectorXd, Eigen::MatrixXd> matlabEigs(const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& areas, const size_t numeigs, const double sigma = 0.0) {
	initMatlabEigs();

	auto [rowIndex, colIndex, nnzValue] = kika::eig::getDataFromSparseMat<size_t>(lbo, 1);
	mwArray A = mwArray::NewSparse(rowIndex.size(), rowIndex.data(), colIndex.size(), colIndex.data(),
		nnzValue.size(), nnzValue.data(), lbo.rows(), lbo.cols(), lbo.nonZeros());
	
	rowIndex.resize(areas.rows());
	colIndex.resize(areas.rows());
	nnzValue.resize(areas.rows());
	for (Eigen::Index i = 0; i < areas.rows(); i++) {
		rowIndex[i] = i + 1;
		colIndex[i] = i + 1;
		nnzValue[i] = areas[i];
	}

	mwArray B = mwArray::NewSparse(rowIndex.size(), rowIndex.data(), colIndex.size(), colIndex.data(),
		nnzValue.size(), nnzValue.data(), lbo.rows(), lbo.cols(), areas.rows());

	mwArray evecs, evals;
	mwArray k = mwArray(numeigs);
	mwArray s = mwArray(sigma);
	small_eigs(2, evecs, evals, A, B, k, s);

	Eigen::MatrixXd retEvecs(lbo.rows(), numeigs);
	evecs.GetData(retEvecs.data(), retEvecs.size());
	Eigen::VectorXd retEvals(numeigs);
	if (evals.IsComplex()) {
		evals.Real().GetData(retEvals.data(), numeigs);
	} else {
		evals.GetData(retEvals.data(), numeigs);
	}
	return std::move(std::make_pair(retEvals, retEvecs));
}

inline std::pair<Eigen::VectorXcd, Eigen::MatrixXcd> matlabEigs(
	const Eigen::SparseMatrix<Eigen::dcomplex>& clbo, const Eigen::VectorXd& areas, const size_t numeigs, const double sigma = 0.0) {
	initMatlabEigs();

	auto [rowIndex, colIndex, nnzReal, nnzImag] = kika::eig::getDataFromSparseMatCmplx<size_t>(clbo, 1);
	mwArray A = mwArray::NewSparse(rowIndex.size(), rowIndex.data(), colIndex.size(), colIndex.data(),
		nnzReal.size(), nnzReal.data(), nnzImag.data(), clbo.rows(), clbo.cols(), clbo.nonZeros());

	rowIndex.resize(areas.rows());
	colIndex.resize(areas.rows());
	nnzReal.resize(areas.rows());
	nnzImag.resize(areas.rows());
	for (Eigen::Index i = 0; i < areas.rows(); i++) {
		rowIndex[i] = i + 1;
		colIndex[i] = i + 1;
		nnzReal[i] = areas[i];
		nnzImag[i] = 0.0;
	}
	
	mwArray B = mwArray::NewSparse(rowIndex.size(), rowIndex.data(), colIndex.size(), colIndex.data(),
		nnzReal.size(), nnzReal.data(), nnzImag.data(), clbo.rows(), clbo.cols(), areas.rows());
	
	mwArray evecs, evals;
	mwArray k = mwArray(numeigs);
	mwArray s = mwArray(sigma);
	small_eigs(2, evecs, evals, A, B, k, s);
	
	Eigen::VectorXd retValue(clbo.rows() * numeigs);
	
	auto evecsReal = evecs.Real(), evecsImag = evecs.Imag();
	Eigen::MatrixXcd retEvecs(clbo.rows(), numeigs);

	evecsReal.GetData(retValue.data(), retValue.size());
	for (size_t i = 0; i < (size_t)retValue.size(); i++) retEvecs.real()(i) = retValue(i);
	evecsImag.GetData(retValue.data(), retValue.size());
	for (size_t i = 0; i < (size_t)retValue.size(); i++) retEvecs.imag()(i) = retValue(i);

	retValue.resize(numeigs);
	Eigen::VectorXcd retEvals(numeigs);
	if (evals.IsComplex()) {
		auto evalsReal = evals.Real(), evalsImag = evals.Imag();
		evalsReal.GetData(retValue.data(), numeigs);
		for (size_t i = 0; i < (size_t)numeigs; i++) retEvals(i).real(retValue(i));
		evalsImag.GetData(retValue.data(), numeigs);
		for (size_t i = 0; i < (size_t)numeigs; i++) retEvals(i).imag(retValue(i));
	} else {
		evals.GetData(retValue.data(), numeigs);
		for (size_t i = 0; i < (size_t)numeigs; i++) {
			retEvals(i).real(retValue(i));
			retEvals(i).imag(0.0);
		}

	}
	return std::move(std::make_pair(retEvals, retEvecs));
}
}

#endif

#if __has_include("localizedManifoldHarmonics.h")
#pragma comment(lib, "localizedManifoldHarmonics")
#include <localizedManifoldHarmonics.h>

namespace kika::mlab {
inline void initLocalManHarmonics() {
	static bool localManHormonicsFirstInitialization = false;
	if (!localManHormonicsFirstInitialization)
		localManHormonicsFirstInitialization = localizedManifoldHarmonicsInitialize();
}

inline std::pair<Eigen::VectorXd, Eigen::MatrixXd> localManHarmonics(
	const Eigen::SparseMatrix<double>& lbo, const Eigen::VectorXd& areas, const Eigen::MatrixXd& evecs,
	const Eigen::VectorXd& region, const size_t numlmh,
	const double regCoeff = 8000.0, const double orthoCoeff = 100000000.0) {
	initLocalManHarmonics();

	Eigen::SparseMatrix<double> A = kika::eig::speye(areas.rows()) * areas.asDiagonal();
	auto [rowIndex, colIndex, nnzValue] = kika::eig::getDataFromSparseMat<size_t>(A, 1);
	mwArray mwAreas = mwArray::NewSparse(rowIndex.size(), rowIndex.data(), colIndex.size(), colIndex.data(),
		nnzValue.size(), nnzValue.data(), A.rows(), A.cols(), nnzValue.size());

	Eigen::SparseMatrix<double> R = lbo + regCoeff * A * region.asDiagonal();
	std::tie(rowIndex, colIndex, nnzValue) = kika::eig::getDataFromSparseMat<size_t>(R, 1);
	mwArray mwA = mwArray::NewSparse(rowIndex.size(), rowIndex.data(), colIndex.size(), colIndex.data(),
		nnzValue.size(), nnzValue.data(), A.rows(), A.cols(), nnzValue.size());

	Eigen::MatrixXd B = A * evecs;
	mwArray mwB = mwArray(B.rows(), B.cols(), mxDOUBLE_CLASS, mxREAL);
	mwB.SetData(B.data(), B.size());

	mwArray k(numlmh);
	mwArray mu(orthoCoeff);
	mwArray lmhvecs, lmhvals;
	localizedManifoldHarmonics(2, lmhvecs, lmhvals, mwA, mwAreas, mwB, k, mu);

	Eigen::MatrixXd retVecs(lbo.rows(), numlmh);
	lmhvecs.GetData(retVecs.data(), retVecs.size());

	Eigen::VectorXd retVals(numlmh);
	lmhvals.GetData(retVals.data(), numlmh);

	return std::move(std::make_pair(retVals, retVecs));
}

// this is test function.
inline std::pair<Eigen::VectorXcd, Eigen::MatrixXcd> localManHarmonics(
	const Eigen::SparseMatrix<Eigen::dcomplex>& clbo, 
	const Eigen::VectorXd& areas,
	const Eigen::MatrixXcd& cevecs,
	const Eigen::VectorXd& region, const size_t numlmh,
	const double regCoeff = 8000.0, 
	const double orthoCoeff = 100000000.0) {

	initLocalManHarmonics();

	Eigen::SparseMatrix<double> mass = kika::eig::speye(areas.rows()) * areas.asDiagonal();
	Eigen::SparseMatrix<Eigen::dcomplex> R = clbo + (regCoeff * mass * region.asDiagonal()).cast<Eigen::dcomplex>();

	auto [rowIndex, colIndex, nnzReal, nnzImag] = kika::eig::getDataFromSparseMatCmplx<size_t>(R, 1);
	mwArray mwA = mwArray::NewSparse(rowIndex.size(), rowIndex.data(), colIndex.size(), colIndex.data(),
		nnzReal.size(), nnzReal.data(), nnzImag.data(), R.rows(), R.cols(), nnzReal.size());
	
	rowIndex.resize(areas.rows());
	colIndex.resize(areas.rows());
	nnzReal.resize(areas.rows());
	nnzImag.resize(areas.rows());
	for (Eigen::Index i = 0; i < areas.rows(); i++) {
		rowIndex[i] = i + 1;
		colIndex[i] = i + 1;
		nnzReal[i] = areas[i];
		nnzImag[i] = 0.0;
	}

	mwArray mwAreas = mwArray::NewSparse(rowIndex.size(), rowIndex.data(), colIndex.size(), colIndex.data(),
		nnzReal.size(), nnzReal.data(), nnzImag.data(), clbo.rows(), clbo.cols(), areas.rows());

	Eigen::MatrixXcd B = mass * cevecs;
	mwArray mwB = mwArray(B.rows(), B.cols(), mxDOUBLE_CLASS, mxCOMPLEX);
	mwB.Real().SetData(B.real().data(), B.size());
	mwB.Imag().SetData(B.imag().data(), B.size());
	
	mwArray k(numlmh);
	mwArray mu(orthoCoeff);
	mwArray lmhvecs, lmhvals;
	localizedManifoldHarmonics(2, lmhvecs, lmhvals, mwA, mwAreas, mwB, k, mu);
	
	Eigen::VectorXd retValue(clbo.rows() * numlmh);

	auto evecsReal = lmhvecs.Real(), evecsImag =lmhvecs.Imag();
	Eigen::MatrixXcd retEvecs(clbo.rows(), numlmh);

	evecsReal.GetData(retValue.data(), retValue.size());
	for (size_t i = 0; i < (size_t)retValue.size(); i++) retEvecs.real()(i) = retValue(i);
	evecsImag.GetData(retValue.data(), retValue.size());
	for (size_t i = 0; i < (size_t)retValue.size(); i++) retEvecs.imag()(i) = retValue(i);

	retValue.resize(numlmh);
	Eigen::VectorXcd retEvals(numlmh);
	if (lmhvals.IsComplex()) {
		auto evalsReal = lmhvals.Real(), evalsImag = lmhvals.Imag();
		evalsReal.GetData(retValue.data(), numlmh);
		for (size_t i = 0; i < (size_t)numlmh; i++) retEvals(i).real(retValue(i));
		evalsImag.GetData(retValue.data(), numlmh);
		for (size_t i = 0; i < (size_t)numlmh; i++) retEvals(i).imag(retValue(i));
	} else {
		lmhvals.GetData(retValue.data(), numlmh);
		for (size_t i = 0; i < (size_t)numlmh; i++) {
			retEvals(i).real(retValue(i));
			retEvals(i).imag(0.0);
		}
	}

	return std::move(std::make_pair(retEvals, retEvecs));
}
}
#endif

#if __has_include("commutFmap.h")
#pragma comment(lib, "commutFmap")
#include <commutFmap.h>
namespace kika::mlab {
inline void initCommutPreserveFmap() {
	static bool commutPreserveFmapFirstInitialization = false;
	if (!commutPreserveFmapFirstInitialization)
		commutPreserveFmapFirstInitialization = commutFmapInitialize();
}

inline Eigen::MatrixXd commutPreserveFmap(
	const Eigen::MatrixXd& oldMap, 
	const Eigen::VectorXd& srcEvals, const Eigen::MatrixXd& srcMH,
	const Eigen::VectorXd& srcAreas, const Eigen::MatrixXd& srcDescs,
	const Eigen::VectorXd& trgEvals, const Eigen::MatrixXd& trgMH,
	const Eigen::VectorXd& trgAreas, const Eigen::MatrixXd& trgDescs,
	const double alpha = 1e-1, const double beta = 1.0, const double gamma = 1e-3) {
	initCommutPreserveFmap();
	
	size_t sk = (size_t)srcEvals.rows(), tk = (size_t)trgEvals.rows();

	mwArray mwSrcEvals(srcEvals.rows(), srcEvals.cols(), mxDOUBLE_CLASS, mxREAL);
	mwSrcEvals.SetData(srcEvals.data(), srcEvals.size());
	mwArray mwSrcMH(srcMH.rows(), srcMH.cols(), mxDOUBLE_CLASS, mxREAL);
	mwSrcMH.SetData(srcMH.data(), srcMH.size());
	mwArray mwSrcDescs(srcDescs.rows(), srcDescs.cols(), mxDOUBLE_CLASS, mxREAL);
	mwSrcDescs.SetData(srcDescs.data(), srcDescs.size());
	Eigen::SparseMatrix<double> A = kika::eig::speye(srcAreas.rows()) * srcAreas.asDiagonal();
	auto [rowIndex, colIndex, nnzValue] = kika::eig::getDataFromSparseMat<size_t>(A, 1);
	mwArray mwSrcAreas = mwArray::NewSparse(rowIndex.size(), rowIndex.data(), colIndex.size(), colIndex.data(),
		nnzValue.size(), nnzValue.data(), A.rows(), A.cols(), nnzValue.size());

	mwArray mwTrgEvals(trgEvals.rows(), trgEvals.cols(), mxDOUBLE_CLASS, mxREAL);
	mwTrgEvals.SetData(trgEvals.data(), trgEvals.size());
	mwArray mwTrgMH(trgMH.rows(), trgMH.cols(), mxDOUBLE_CLASS, mxREAL);
	mwTrgMH.SetData(trgMH.data(), trgMH.size());
	mwArray mwTrgDescs(trgDescs.rows(), trgDescs.cols(), mxDOUBLE_CLASS, mxREAL);
	mwTrgDescs.SetData(trgDescs.data(), trgDescs.size());
	A = kika::eig::speye(trgAreas.rows()) * trgAreas.asDiagonal();
	std::tie(rowIndex, colIndex, nnzValue) = kika::eig::getDataFromSparseMat<size_t>(A, 1);
	mwArray mwTrgAreas = mwArray::NewSparse(rowIndex.size(), rowIndex.data(), colIndex.size(), colIndex.data(),
		nnzValue.size(), nnzValue.data(), A.rows(), A.cols(), nnzValue.size());

	mwArray init(sk, tk, mxDOUBLE_CLASS, mxREAL);
	init.SetData(oldMap.data(), oldMap.size());

	mwArray fmap;
	mwArray srck(sk), trgk(tk);
	mwArray a(alpha), b(beta), c(gamma);
	commutFmap(1, fmap, init, mwSrcMH, mwTrgMH,
		mwSrcEvals, mwSrcEvals, mwSrcAreas,
		mwTrgAreas, mwSrcDescs, mwTrgDescs,
		srck, trgk, a, b, c);

	Eigen::MatrixXd retFmap(tk, sk);
	fmap.GetData(retFmap.data(), retFmap.size());
	
	return retFmap;
}
}
#endif