#pragma once

/// This method is not suitable for geodesic heat.
/// because sparse rhs can not be prolonged properly.
 
/*
GravoMG::MultigridSolver solver(V, F, M);
solver.buildHierarchy();
solver.factorizationType = 1;
solver.isSmootherGaussSeidel = true;
std::cout << solver.maxIter << std::endl;
std::cout << solver.samples.size() << std::endl;

size_t nv = mesh.getNumVerts(), nf = mesh.getNumFaces();

Eigen::VectorXd delta = Eigen::VectorXd::Zero(nv);
delta(0) = 1.0;

double lambda = kika::util::pow2(mesh.maxDiagLength());

Eigen::MatrixXd u = Eigen::MatrixXd::Zero(nv, 1);
Eigen::SparseMatrix<double> heat = kika::eig::spdiag(dsl.vertexAreas()) + lambda * dsl.laplacianOp();
Eigen::MatrixXd D = delta;
solver.solve(heat, D, u);

Eigen::MatrixX3d x = -dsl.integratedGradient(u.col(0));

Eigen::VectorXd dx = dsl.divergenceOp() * x;

Eigen::MatrixXd Dx = -dx;
Eigen::MatrixXd Dist = Eigen::MatrixXd::Zero(nv, 1);
Eigen::SparseMatrix<double> poisson = -dsl.laplacianOp();
solver.solve(poisson, Dx, Dist);
Eigen::VectorXd distance = Dist.col(0);
distance.array() -= distance.minCoeff();

auto color = kika::color::normalizedColor(distance, kika::color::palette::whiteBlack());
color = kika::color::waveColor(color);
*/
/// 
/// 

////
//// MIT License
//// 
//// Copyright(c) 2023 rubenwiersma
//// 
//// Permission is hereby granted, free of charge, to any person obtaining a copy
//// of this software and associated documentation files(the "Software"), to deal
//// in the Software without restriction, including without limitation the rights
//// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
//// copies of the Software, and to permit persons to whom the Software is
//// furnished to do so, subject to the following conditions :
//// 
//// The above copyright notice and this permission notice shall be included in all
//// copies or substantial portions of the Software.
//// 
//// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
//// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//// SOFTWARE.
//// 
//// A Fast Geometric Multigrid Method for Curved Surfaces. Wiersma, R. et al.
//// https://github.com/rubenwiersma/gravo_mg_cpp
////

#define _USE_MATH_DEFINES

#include <cmath>
#include <numeric>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <set>
#include <vector>
#include <random>

#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>

using namespace std;


namespace GravoMG {
	void scaleMesh(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double scaleRatio = 1.0)
	{
		//cout << "> Scaling the mesh to have a unit length \n";
		Eigen::Vector3d minV;
		Eigen::Vector3d maxV;
		Eigen::Vector3d length;
		Eigen::MatrixXd MatSubs(V.rows(), V.cols());
		Eigen::MatrixXd MatAdd(V.rows(), V.cols());
		double maxVal;
		double scalingRatio = scaleRatio; //dia-meter, not radius


		/* Get the min and max coefficients */
		for (int i = 0; i < V.cols(); i++)
		{
			minV(i) = V.col(i).minCoeff();
			maxV(i) = V.col(i).maxCoeff();
			length(i) = maxV(i) - minV(i);
			MatSubs.col(i).setConstant(minV(i));
		}

		maxVal = length.maxCoeff();

		/* Translate to the Origin */
		V = V - MatSubs;

		/* Scale w.r.t the longest axis */
		V = V * (scalingRatio / maxVal);

		for (int i = 0; i < V.cols(); i++) {
			maxV(i) = V.col(i).maxCoeff();
			MatAdd.col(i).setConstant(0.5 * maxV(i));
		}

		/* Translate s.t. the center is in the Origin */
		V = V - MatAdd;

		for (int i = 0; i < V.cols(); i++)
		{
			minV(i) = V.col(i).minCoeff();
			maxV(i) = V.col(i).maxCoeff();
			length(i) = maxV(i) - minV(i);
		}
		maxVal = length.maxCoeff();
	}

}

#include <queue>
#include <vector>

/* Data structure for Priority Queue */
struct VertexPair {
	int		vId;
	double	distance;
	bool	operator> (const VertexPair& ref) const { return distance > ref.distance; }
	bool	operator< (const VertexPair& ref) const { return distance < ref.distance; }
};

/* Enum to set solver type */
enum Sampling {
	FASTDISK = 0,
	RANDOM = 1,
	MIS = 2
};

enum Weighting {
	BARYCENTRIC = 0,
	UNIFORM = 1,
	INVDIST = 2
};

namespace GravoMG {
	class MultigridSolver
	{
	public:
		MultigridSolver(Eigen::MatrixXd& V, Eigen::MatrixXi& neigh, Eigen::SparseMatrix<double>& M);
		~MultigridSolver();
		/* Hierarchy-related methods */
		void buildHierarchy();
		void constructProlongation();
		double computeAverageEdgeLength(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& neigh);
		double inTriangle(const Eigen::RowVector3d& p, const std::vector<int>& tri, const Eigen::RowVector3d& triNormal, const Eigen::MatrixXd& pos, Eigen::RowVector3d& bary, std::map<int, float>& insideEdge);
		std::vector<double> uniformWeights(const int& n_points);
		std::vector<double> inverseDistanceWeights(const Eigen::MatrixXd& pos, const Eigen::RowVector3d& source, const std::vector<int>& edges);
		std::vector<int> maximumDeltaIndependentSet(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& edges, const double& radius);
		std::vector<int> maximumDeltaIndependentSet(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& edges, const double& radius, Eigen::VectorXd& D, std::vector<size_t>& nearestSourceK);
		std::vector<int> fastDiskSample(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& edges, const double& radius, Eigen::VectorXd& D, std::vector<size_t>& nearestSourceK);
		void constructDijkstraWithCluster(const Eigen::MatrixXd& points, const std::vector<int>& source, const Eigen::MatrixXi& neigh, int k, Eigen::VectorXd& D, std::vector<size_t>& nearestSourceK);

		// Multigrid solvers	
		// Using Gauss-Seidel Smoother (our main process)
		double multiGridVCycleGS(Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& b, Eigen::MatrixXd& x, int k, bool isDebug = true);
		double multiGridFCycleGS(Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& b, Eigen::MatrixXd& x, int k, bool isDebug = true);
		double multiGridWCycleGS(Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& b, Eigen::MatrixXd& x, int k, bool isDebug = true);
		void GaussSeidelSmoother(Eigen::SparseMatrix<double>& LHS, Eigen::MatrixXd& rhs,
			Eigen::MatrixXd& x, int maxIter, double tol, bool isDebug = true);

		double residualCheck(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& b, const Eigen::MatrixXd& x, int type);

		/* Core solver function */
		void solve(Eigen::SparseMatrix<double>& LHS, Eigen::MatrixXd& rhs, Eigen::MatrixXd& x);

		/* Position-related variables */
		Eigen::MatrixXd						V;							//!< Coordinates of each vertex
		Eigen::MatrixXd						V0;							//!< Original coordinates of each vertex
		Eigen::MatrixXd						normals;
		Eigen::SparseMatrix<double>			M;							//!<mass matrix to use>
		Eigen::SparseMatrix<double>			Minv;							//!<inverse mass matrix to use>
		std::vector<Eigen::MatrixXd> 		levelV;
		std::vector<Eigen::MatrixXi> 		levelE;
		std::vector<std::vector<int>>		noTriFoundMap;
		std::vector<std::vector<std::vector<int>>>		allTriangles;
		std::vector<Eigen::MatrixXd>		levelN;

		/* Hierarchy data*/
		std::vector<size_t>								DoF;							//!< Degrees of freedom per level
		std::vector<Eigen::SparseMatrix<double>>		U;								//!< Prolongation operator
		std::vector<Eigen::SparseMatrix<double>>		UNeigh;
		std::vector<Eigen::SparseMatrix<double>>		Abar, Sbar, Mbar;				//!< Reduced system matrices
		std::vector<Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper>> cgSolvers;
		std::vector<size_t>								sampleIDHierarchy;
		std::vector<Eigen::MatrixXi>					neighHierarchy;
		std::vector<std::vector<size_t>>				nearestSource;                  // the shortest-distanced source from a given point
		std::vector<Eigen::VectorXd>					geodDistance;                   // Geodesic distance on each level (based on the number of samples on each level)
		int												geodesicType = 0;                 // Type of the hierarchy disk: 0 => HSIM-type || 1 => GVD-type (non-overlap) || 2 => combinatorial (only points in kNN)
		int												prolongationFunction = 0;			// Type of mapping from geodesic distance to prolongation operator value. 0 => constant, 1 => linear, 2 => 5th order polynomial,  3 => 2nd order polynomial
		std::vector<std::vector<int>>					samples;
		std::vector<size_t>								PointsToSampleMap;
		std::vector<std::vector<size_t>>				PointsToSampleMaps;
		int									cycleType = 0;								//0: V-cycle, 1: F-cycle, 2: W-cycle
		int									factorizationType = 0;						//0: LLT,	1: LDLT  
		bool                                isSmootherGaussSeidel;
		bool                                isGVDTwoRings;
		bool								isClusterExpanded;
		bool                                isPartitionOfUnity;
		bool								isUsingLocalMax;
		bool								nested = false;								// Whether to use a nested hierarchy or move the coarse points
		int									knnNeigh;
		int									stoppingCriteria = 0;						//!< Types of norm for residual check => 0: rel. norm (||Ax-b||/||b||)		1: L2 M^-1 (||Ax-b||_{M-1}/||b||_{M-1})		2: L2 M (||Ax-b||{M}/||b||_{M})		3: Abs (Ax-b).norm()
		int									maxIter = 50;
		int									lowBound = 1000;
		double								ratio = 8;									// The fraction of points to keep in each level: 1 / ratio
		Sampling							samplingStrategy = FASTDISK;				// Which sampling strategy to use. Options are: FASTDISK, POISSONDISK, FPS, RANDOM, MIS
		Weighting							weightingScheme = BARYCENTRIC;				// Which weighting scheme to use. Options are: BARYCENTRIC, UNIFORM, INVDIST

		Eigen::MatrixXi						neigh;										//!< Neighboring informaiton of the lowest level

		/* Solver-related data */
		int preIters;
		int postIters;
		double accuracy = 5e-4;
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> coarsestSolver;
		bool POUtruncated = true;

		/* Dummy data*/
		Eigen::MatrixXd						sampledPoints;
		Eigen::MatrixXd						Vplanar;
		Eigen::MatrixXi						Fplanar;
		int hierarchyType;							//0: non-nested, 1:GVDE
		std::vector<Eigen::MatrixXd>		newCentroidHierarchy;

		/* Ablation parameters */
		bool								ablation = false;
		int									ablationNumPoints = 3;
		bool								ablationRandom = false;

	private:

	};


}


namespace GravoMG {
	/* Constructor */
	MultigridSolver::MultigridSolver(
		Eigen::MatrixXd& V, Eigen::MatrixXi& neigh, Eigen::SparseMatrix<double>& M) :
		V(V), neigh(neigh), M(M) {
		Minv = (Eigen::SparseMatrix<double>) M.diagonal().cwiseInverse().asDiagonal();

		V0 = V;
	}

	/* ====================== BASIC FUNCTIONS  ======================= */
	MultigridSolver::~MultigridSolver()
	{
		// std::cout << "Deleting data!\n";
		V.resize(0, 0);
		V0.resize(0, 0);
		neigh.resize(0, 0);
		DoF.clear(); DoF.shrink_to_fit();
		U.clear(); U.shrink_to_fit();
		Abar.clear(); Abar.shrink_to_fit();
		Mbar.clear(); Mbar.shrink_to_fit();
		neighHierarchy.clear(); neighHierarchy.shrink_to_fit();
		nearestSource.clear(); nearestSource.shrink_to_fit();
		geodDistance.clear(); geodDistance.shrink_to_fit();
		samples.clear(); samples.shrink_to_fit();
		PointsToSampleMap.clear(); PointsToSampleMap.shrink_to_fit();
	}

	// Build the hierarchy
	void MultigridSolver::buildHierarchy()
	{
		nearestSource.clear();
		nearestSource.shrink_to_fit();

		constructProlongation();
	}

	void MultigridSolver::constructProlongation()
	{
		// Create prolongation operator for level k+1 to k
		// Points in current level
		Eigen::MatrixXd levelPoints = V;
		Eigen::MatrixXd levelNormals = normals;
		// Points in coarser level
		Eigen::MatrixXd samplePoints;
		// Neighborhood data structure
		Eigen::MatrixXi neighLevelK = neigh;

		// Compute initial radius
		double densityRatio = std::sqrt(ratio);
		int nLevel1 = int(levelPoints.rows() / ratio);

		std::random_device					rd;
		std::default_random_engine			generator(rd());

		// For each level
		int k = 0;
		DoF.clear();
		DoF.shrink_to_fit();
		DoF.push_back(levelPoints.rows());
		while (levelPoints.rows() > lowBound && k < 10) {
			double radius = std::cbrt(ratio) * computeAverageEdgeLength(levelPoints, neighLevelK);
			// -- Setting up variables
			// Data structure for neighbors inside level k
			std::vector<std::set<int>> neighborsList;

			// List of triplets to build prolongation operator U
			std::vector<Eigen::Triplet<double>> AllTriplet, UNeighAllTriplet;

			// The nearest coarse point for each fine point and the distances computed
			Eigen::VectorXd D(levelPoints.rows());
			D.setConstant(std::numeric_limits<double>::max());
			nearestSource.push_back(std::vector<size_t>(levelPoints.rows()));

			switch (samplingStrategy) {
			case FASTDISK:
				samples.push_back(fastDiskSample(levelPoints, neighLevelK, radius, D, nearestSource[k]));
				DoF.push_back(samples[k].size());
				break;
			case RANDOM:
				DoF.push_back(DoF[k] / ratio);
				samples.push_back(std::vector<int>(DoF[k]));
				std::iota(samples[k].begin(), samples[k].end(), 0);
				std::shuffle(samples[k].begin(), samples[k].end(), generator);
				samples[k].resize(DoF[k + 1]);
				break;
			case MIS:
				samples.push_back(maximumDeltaIndependentSet(levelPoints, neighLevelK, radius, D, nearestSource[k]));
				DoF.push_back(samples[k].size());
				break;
			}

			if (samples[k].size() < lowBound) {
				nearestSource.pop_back();
				break;
			}

			DoF[k + 1] = samples[k].size();

			// -- Compute distance from fine points to coarse points and get closest coarse point
			// using distances from MIS if computed before
			constructDijkstraWithCluster(levelPoints, samples[k], neighLevelK, k, D, nearestSource[k]); // Stores result in nearestSource[k]

			// Create neighborhood for the next level
			neighborsList.resize(DoF[k + 1]);
			for (int fineIdx = 0; fineIdx < DoF[k]; ++fineIdx) {
				for (int j = 0; j < neighLevelK.cols(); ++j) {
					int neighIdx = neighLevelK(fineIdx, j);
					if (neighIdx < 0) break;
					if (nearestSource[k][fineIdx] != nearestSource[k][neighIdx]) {
						neighborsList[nearestSource[k][fineIdx]].insert(nearestSource[k][neighIdx]);
					}
				}
			}

			// Store in homogeneous datastructure
			int maxNeighNum = 0;
			for (int i = 0; i < neighborsList.size(); ++i) {
				if (neighborsList[i].size() > maxNeighNum) {
					maxNeighNum = neighborsList[i].size();
				}
			}
			neighLevelK.resize(DoF[k + 1], maxNeighNum);
			neighLevelK.setConstant(-1);
			for (int i = 0; i < neighborsList.size(); ++i) {
				neighLevelK(i, 0) = i;
				int iCounter = 1;
				for (int node : neighborsList[i]) {
					if (node == i)	continue;
					if (iCounter >= maxNeighNum) break;
					neighLevelK(i, iCounter) = node;
					iCounter++;
				}
			}

			// Setting up the DoF for the next level
			// tempPoints are the centers of the voronoi cells, each row for each voronoi cells
			Eigen::MatrixXd tempPoints(DoF[k + 1], levelPoints.cols());
			tempPoints.setZero();
			if (nested) {
				for (int coarseIdx = 0; coarseIdx < DoF[k + 1]; ++coarseIdx) {
					tempPoints.row(coarseIdx) = levelPoints.row(samples[k][coarseIdx]);
				}
			} else {
				std::vector<int> clusterSizes(DoF[k + 1]);
				for (int fineIdx = 0; fineIdx < DoF[k]; ++fineIdx) {
					int coarseIdx = nearestSource[k][fineIdx];
					tempPoints.row(coarseIdx) = tempPoints.row(coarseIdx) + levelPoints.row(fineIdx);
					++clusterSizes[coarseIdx];
				}
				for (int coarseIdx = 0; coarseIdx < DoF[k + 1]; ++coarseIdx) {
					if (clusterSizes[coarseIdx] == 1) {
						tempPoints.row(coarseIdx) = levelPoints.row(samples[k][coarseIdx]);
						for (int neighIdx : neighborsList[coarseIdx]) {
							tempPoints.row(coarseIdx) = tempPoints.row(coarseIdx) + levelPoints.row(samples[k][neighIdx]);
						}
						tempPoints.row(coarseIdx) = tempPoints.row(coarseIdx) / (neighborsList[coarseIdx].size() + 1.);
					} else {
						tempPoints.row(coarseIdx) = tempPoints.row(coarseIdx) / clusterSizes[coarseIdx];
					}
				}
			}

			// Create triangles for this level based on Voronoi cells
			std::vector<std::vector<int>> tris;
			tris.reserve(DoF[k + 1] * maxNeighNum);
			std::vector<std::vector<int>> connectedTris(DoF[k + 1]);
			std::vector<Eigen::RowVector3d> triNormals;
			triNormals.reserve(DoF[k + 1] * maxNeighNum);
			int triIdx = 0;
			for (int coarseIdx = 0; coarseIdx < DoF[k + 1]; ++coarseIdx) {
				// Iterate over delaunay triangles
				int v2Idx, v3Idx;
				for (auto it = neighborsList[coarseIdx].begin(); it != neighborsList[coarseIdx].end(); it++) {
					v2Idx = *it;
					// We iterate over the coarse indices in order,
					// so if the neighboring idx is lower then the current coarseIdx,
					// it must have been considered before and be part of a triangle.
					if (v2Idx < coarseIdx) continue;
					for (auto it2 = std::next(it); it2 != neighborsList[coarseIdx].end(); it2++) {
						v3Idx = *it2;
						if (v3Idx < coarseIdx) continue;
						if (neighborsList[v2Idx].find(v3Idx) != neighborsList[v2Idx].end()) {
							tris.push_back({ coarseIdx, v2Idx, v3Idx });
							Eigen::RowVector3d e12 = tempPoints.row(v2Idx) - tempPoints.row(coarseIdx);
							Eigen::RowVector3d e13 = tempPoints.row(v3Idx) - tempPoints.row(coarseIdx);
							triNormals.push_back(e12.cross(e13).normalized());
							connectedTris[coarseIdx].push_back(triIdx);
							connectedTris[v2Idx].push_back(triIdx);
							connectedTris[v3Idx].push_back(triIdx);
							++triIdx;
						}
					}
				}
			}
			tris.shrink_to_fit();
			triNormals.shrink_to_fit();

			// Create local triangulation on each cluster (centralized at sample i)
			int notrisfound = 0;
			int edgesfound = 0;
			int fallbackCount = 0;

			for (int fineIdx = 0; fineIdx < DoF[k]; ++fineIdx) {
				Eigen::RowVector3d finePoint = levelPoints.row(fineIdx);
				int coarseIdx = nearestSource[k][fineIdx];
				Eigen::RowVector3d coarsePoint = tempPoints.row(coarseIdx);
				std::vector<double> weights;

				if (nested && samples[k][nearestSource[k][fineIdx]] == fineIdx) {
					AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, 1.));
					continue;
				}

				if (neighborsList[coarseIdx].empty()) {
					// If the coarse point has no neighbors,
					// set the weight to 1 for the coarse point.
					// Note: this should not happen.
					AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, 1.));
				} else if (neighborsList[coarseIdx].size() == 1) {
					// If the coarse point only has one neighbor, no triangle can be created.
					// Thus, the weights are distributed w.r.t. the distance to each coarse point.
					int neighIdx = *neighborsList[coarseIdx].begin();
					Eigen::RowVector3d neighPoint = tempPoints.row(neighIdx);

					// get the distance to the two neighboring centroids
					Eigen::RowVector3d e12 = neighPoint - coarsePoint;
					double e12Length = max(e12.norm(), 1e-8);
					double w2 = (levelPoints.row(fineIdx) - coarsePoint).dot(e12.normalized()) / e12Length;
					w2 = min(max(w2, 0.), 1.);
					double w1 = 1 - w2;

					switch (weightingScheme) {
					case BARYCENTRIC:
						AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, w1));
						AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, neighIdx, w2));
						break;
					case UNIFORM:
						weights = uniformWeights(2);
						AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, weights[0]));
						AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, neighIdx, weights[1]));
						break;
					case INVDIST:
						std::vector<int> endPoints = { coarseIdx, neighIdx };
						weights = inverseDistanceWeights(tempPoints, finePoint, endPoints);
						AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, weights[0]));
						AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, neighIdx, weights[1]));
						break;
					}
				} else {
					// Only keep triangle with minimum distance
					double minDistToTriangle = std::numeric_limits<double>::max();
					Eigen::RowVector3d minBary = { 0., 0., 0. };
					std::vector<int> minTri;
					bool triFound = false;

					// Values are positive if inside and negative if not
					// Float value represents distance
					std::map<int, float> insideEdge;

					// Iterate over all triangles
					for (int triIdx : connectedTris[coarseIdx]) {
						std::vector<int> tri = tris[triIdx];
						// Make sure that the coarseIdx is in position 0, while keeping orientation
						while (tri[0] != coarseIdx) std::rotate(tri.begin(), tri.begin() + 1, tri.end());

						Eigen::RowVector3d bary = { 0., 0., 0. };
						// If the triangle contains the point, the distance is positive, else it's negative
						double distToTriangle = inTriangle(finePoint, tri, triNormals[triIdx], tempPoints, bary, insideEdge);
						if (distToTriangle >= 0. && distToTriangle < minDistToTriangle) {
							triFound = true;
							minDistToTriangle = distToTriangle;
							minTri = tri;
							minBary = bary;
							break;
						}
					}

					if (triFound) {
						switch (weightingScheme) {
						case BARYCENTRIC:
							AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[0], minBary(0)));
							AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[1], minBary(1)));
							AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[2], minBary(2)));
							break;
						case UNIFORM:
							weights = uniformWeights(3);
							AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[0], weights[0]));
							AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[1], weights[1]));
							AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[2], weights[2]));
							break;
						case INVDIST:
							weights = inverseDistanceWeights(tempPoints, finePoint, minTri);
							AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[0], weights[0]));
							AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[1], weights[1]));
							AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[2], weights[2]));
							break;
						}
					} else {
						bool edgeFound = false;
						double minEdge = std::numeric_limits<double>::max();
						int minEdgeIdx = 0;
						for (const auto& element : insideEdge) {
							const auto& key = element.first;
							const auto& value = element.second;
							if (value >= 0. && value < minEdge) {
								edgeFound = true;
								minEdge = value;
								minEdgeIdx = key;
								break;
							}
						}
						if (edgeFound) {
							++edgesfound;
							Eigen::RowVector3d p2 = tempPoints.row(minEdgeIdx);
							Eigen::RowVector3d e12 = p2 - coarsePoint;
							double e12Length = max(e12.norm(), 1e-8);
							double w2 = (finePoint - coarsePoint).dot(e12.normalized()) / e12Length;
							w2 = min(max(w2, 0.), 1.);
							double w1 = 1. - w2;

							switch (weightingScheme) {
							case BARYCENTRIC:
								AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, w1));
								AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minEdgeIdx, w2));
								break;
							case UNIFORM:
								weights = uniformWeights(2);
								AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, weights[0]));
								AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minEdgeIdx, weights[1]));
								break;
							case INVDIST:
								std::vector<int> endPointsEdge = { coarseIdx, minEdgeIdx };
								weights = inverseDistanceWeights(tempPoints, finePoint, endPointsEdge);
								AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, weights[0]));
								AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minEdgeIdx, weights[1]));
								break;
							}
						} else {
							// Use closest three
							std::vector<int> prolongFrom(3);
							prolongFrom[0] = coarseIdx;

							std::vector<VertexPair> pointsDistances;
							for (int j = 0; j < neighLevelK.cols(); ++j) {
								int neighIdx = neighLevelK(coarseIdx, j);
								if (neighIdx < 0 || neighIdx == coarseIdx) continue;
								VertexPair vp = { neighIdx, (finePoint - tempPoints.row(neighIdx)).norm() };
								pointsDistances.push_back(vp);
							}
							std::sort(pointsDistances.begin(), pointsDistances.end(), std::less<VertexPair>());
							for (int j = 1; j < 3; ++j) {
								prolongFrom[j] = pointsDistances[j - 1].vId;
							}
							std::vector<double> weights = inverseDistanceWeights(tempPoints, finePoint, prolongFrom);
							for (int j = 0; j < prolongFrom.size(); j++) {
								AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, prolongFrom[j], weights[j]));
							}
							++fallbackCount;
						}
						++notrisfound;
					}
				}
			}

			levelPoints = tempPoints;

			Eigen::SparseMatrix<double> ULevel;
			ULevel.resize(DoF[k], DoF[k + 1]);
			ULevel.setFromTriplets(AllTriplet.begin(), AllTriplet.end());
			U.push_back(ULevel);
			AllTriplet.clear();
			AllTriplet.shrink_to_fit();
			++k;
		}
	}

	double MultigridSolver::inTriangle(const Eigen::RowVector3d& p, const std::vector<int>& tri, const Eigen::RowVector3d& triNormal, const Eigen::MatrixXd& pos, Eigen::RowVector3d& bary, std::map<int, float>& insideEdge)
	{
		Eigen::RowVector3d v1, v2, v3;
		v1 = pos.row(tri[0]);
		v2 = pos.row(tri[1]);
		v3 = pos.row(tri[2]);
		Eigen::RowVector3d v1ToP = p - v1;
		Eigen::RowVector3d e12 = v2 - v1;
		Eigen::RowVector3d e13 = v3 - v1;

		double distToTriangle = (p - v1).dot(triNormal);
		Eigen::RowVector3d pProjected = p - distToTriangle * triNormal;

		double doubleArea = (v2 - v1).cross(v3 - v1).dot(triNormal);
		bary(0) = (v3 - v2).cross(pProjected - v2).dot(triNormal) / doubleArea;
		bary(1) = (v1 - v3).cross(pProjected - v3).dot(triNormal) / doubleArea;
		bary(2) = 1. - bary(0) - bary(1);

		if (insideEdge.find(tri[1]) == insideEdge.end()) {
			insideEdge[tri[1]] = ((v1ToP)-((v1ToP).dot(e12) * (e12))).norm();
		}
		if (insideEdge.find(tri[2]) == insideEdge.end()) {
			insideEdge[tri[2]] = ((v1ToP)-((v1ToP).dot(e13) * (e13))).norm();
		}
		if (bary(0) < 0. || bary(1) < 0.) {
			insideEdge[tri[1]] = -1.;
		}
		if (bary(0) < 0. || bary(2) < 0.) {
			insideEdge[tri[2]] = -1.;
		}

		if (bary(0) >= 0. && bary(1) >= 0. && bary(2) >= 0.) {
			return abs(distToTriangle);
		}

		return -1.;
	}

	std::vector<double> MultigridSolver::uniformWeights(const int& n_points) {
		std::vector<double> weights(n_points);
		std::fill(weights.begin(), weights.end(), 1. / n_points);
		return weights;
	}

	std::vector<double> MultigridSolver::inverseDistanceWeights(const Eigen::MatrixXd& pos, const Eigen::RowVector3d& p, const std::vector<int>& edges) {
		double sumWeight = 0.;
		std::vector<double> weights(edges.size());
		for (int j = 0; j < edges.size(); ++j) {
			weights[j] = 1. / max(1e-8, (p - pos.row(edges[j])).norm());
			sumWeight += weights[j];
		}
		for (int j = 0; j < weights.size(); ++j) {
			weights[j] = weights[j] / sumWeight;
		}
		return weights;
	}

	double MultigridSolver::computeAverageEdgeLength(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& neigh) {
		double sumLength = 0;
		int nEdges = 0;
		for (int i = 0; i < pos.rows(); ++i) {
			Eigen::Vector3d p1 = pos.row(i);
			for (int j = 0; j < neigh.cols(); ++j) {
				if (neigh(i, j) < 0) continue;
				Eigen::Vector3d p2 = pos.row(neigh(i, j));
				double dist = (p1 - p2).norm();
				if (dist > 0) {
					sumLength += dist;
					++nEdges;
				}
			}
		}
		return sumLength / (double)nEdges;
	}

	std::vector<int> MultigridSolver::maximumDeltaIndependentSet(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& edges, const double& radius) {
		std::vector<bool> visited(edges.rows());
		std::vector<int> selection;
		for (int i = 0; i < edges.rows(); ++i) {
			if (!visited[i]) {
				selection.push_back(i);
				for (int j = 0; j < edges.cols(); ++j) {
					int neighIdx = edges(i, j);
					if (neighIdx < 0) break;
					double dist = (pos.row(i) - pos.row(neighIdx)).norm();
					if (dist < radius) {
						visited[neighIdx] = true;
					}
				}
			}
		}
		return selection;
	}

	std::vector<int> MultigridSolver::maximumDeltaIndependentSet(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& edges, const double& radius, Eigen::VectorXd& D, std::vector<size_t>& nearestSourceK) {
		std::vector<bool> visited(edges.rows());
		std::vector<int> selection;
		int sampleIdx = 0;
		for (int i = 0; i < edges.rows(); ++i) {
			if (!visited[i]) {
				selection.push_back(i);
				nearestSourceK[i] = sampleIdx;
				for (int j = 0; j < edges.cols(); ++j) {
					int neighIdx = edges(i, j);
					if (neighIdx < 0) break;
					double dist = (pos.row(i) - pos.row(neighIdx)).norm();
					if (dist < radius) {
						visited[neighIdx] = true;
						if (dist < D(neighIdx)) {
							D(neighIdx) = dist;
							nearestSourceK[neighIdx] = sampleIdx;
						}
					}
				}
				++sampleIdx;
			}
		}
		return selection;
	}

	std::vector<int> MultigridSolver::fastDiskSample(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& edges, const double& radius, Eigen::VectorXd& D, std::vector<size_t>& nearestSourceK) {
		std::vector<bool> visited(edges.rows());
		std::vector<int> selection;
		int sampleIdx = 0;
		for (int i = 0; i < edges.rows(); ++i) {
			if (!visited[i]) {
				selection.push_back(i);
				nearestSourceK[i] = sampleIdx;
				for (int j = 0; j < edges.cols(); ++j) {
					int neighIdx = edges(i, j);
					if (neighIdx < 0) break;
					double dist = (pos.row(i) - pos.row(neighIdx)).norm();
					if (dist < radius) {
						visited[neighIdx] = true;
						if (dist < D(neighIdx)) {
							D(neighIdx) = dist;
							nearestSourceK[neighIdx] = sampleIdx;
						}
						for (int j2 = 0; j2 < edges.cols(); ++j2) {
							int neighIdx2 = edges(neighIdx, j2);
							if (neighIdx2 < 0) break;
							double dist2 = dist + (pos.row(neighIdx) - pos.row(neighIdx2)).norm();
							if (dist2 < radius) {
								visited[neighIdx2] = true;
								if (dist2 < D(neighIdx2)) {
									D(neighIdx2) = dist2;
									nearestSourceK[neighIdx2] = sampleIdx;
								}
							}
						}
					}
				}
				++sampleIdx;
			}
		}
		return selection;
	}

	void MultigridSolver::constructDijkstraWithCluster(const Eigen::MatrixXd& points, const std::vector<int>& source, const Eigen::MatrixXi& neigh, int k, Eigen::VectorXd& D, std::vector<size_t>& nearestSourceK)
	{
		std::priority_queue<VertexPair, std::vector<VertexPair>, std::greater<VertexPair>> DistanceQueue;
		if (nearestSourceK.empty())	nearestSourceK.resize(points.rows(), source[0]);

		for (int i = 0; i < source.size(); ++i) {
			D(source[i]) = 0.0;
			VertexPair vp{ source[i], D(source[i]) };
			DistanceQueue.push(vp);
			nearestSourceK[source[i]] = i;
		}

		int curSource;
		while (!DistanceQueue.empty()) {
			VertexPair vp1 = DistanceQueue.top();
			curSource = nearestSourceK[vp1.vId];
			Eigen::RowVector3d vertex1 = points.row(vp1.vId);
			DistanceQueue.pop();

			//for (int vNeigh : neigh.row(vp1.vId)) {
			for (int i = 0; i < neigh.cols(); ++i) {
				int vNeigh = neigh(vp1.vId, i);

				if (vNeigh >= 0) {
					double dist, distTemp;
					Eigen::RowVector3d vertex2 = points.row(vNeigh);
					dist = (vertex2 - vertex1).norm();
					distTemp = vp1.distance + dist;
					if (distTemp < D(vNeigh)) {
						// Assign a new distance
						D(vNeigh) = distTemp;
						VertexPair v2{ vNeigh, distTemp };
						DistanceQueue.push(v2);


						// Assign the nearest source to a certain point
						nearestSourceK[vNeigh] = curSource;
					}
				}
			}
		}
	}


	double  MultigridSolver::multiGridVCycleGS(Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& b, Eigen::MatrixXd& x, int k, bool isDebug)
	{
		double tolPre = 1e-15, tolPost = 1e-15;
		//[1] Smoothing
		GaussSeidelSmoother(A, b, x, preIters, tolPre, isDebug);

		//[2] Residual
		Eigen::MatrixXd res = b - A * x;

		//[3] Restriction
		Eigen::MatrixXd resRest = U[k].transpose() * res;

		//[4] Recursion
		Eigen::MatrixXd eps;
		eps.setZero(resRest.rows(), resRest.cols());
		if (k == U.size() - 1) {
			eps = coarsestSolver.solve(resRest);
		} else {
			multiGridVCycleGS(Abar[k + 1], resRest, eps, k + 1, isDebug);
		}

		//[5] Prolongation
		x = x + U[k] * eps;

		//[6] Post-Smoothing
		GaussSeidelSmoother(A, b, x, postIters, tolPost, isDebug);

		return 0.0;
	}

	//!< Perform Multigrid using F-Cycle and Gauss-Seidel smoothing
	double  MultigridSolver::multiGridFCycleGS(Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& b, Eigen::MatrixXd& x, int k, bool isDebug)
	{
		double tolPre = 1e-15, tolPost = 1e-15;
		//[1] Smoothing
		GaussSeidelSmoother(A, b, x, preIters, tolPre, isDebug);

		//[2] Residual
		Eigen::MatrixXd res = b - A * x;

		//[3] Restriction
		Eigen::MatrixXd resRest = U[k].transpose() * res;

		//[4] Recursion
		Eigen::MatrixXd eps;
		eps.setZero(resRest.rows(), resRest.cols());
		if (k == U.size() - 1) {
			eps = coarsestSolver.solve(resRest);
		} else {
			multiGridFCycleGS(Abar[k + 1], resRest, eps, k + 1, isDebug);
		}

		//[5] Prolongation
		x = x + U[k] * eps;

		//[6] Post-Smoothing
		GaussSeidelSmoother(A, b, x, postIters, tolPost, isDebug);

		//[7] Residual
		res = b - A * x;

		//[8] Restriction
		resRest = U[k].transpose() * res;

		//[9] Recursion
		if (k == DoF.size() - 2) {
			eps = coarsestSolver.solve(resRest);
		} else {
			multiGridVCycleGS(Abar[k + 1], resRest, eps, k + 1, isDebug);
		}

		//[10] Prolongation
		x = x + U[k] * eps;

		//[11] Post-Smoothing
		GaussSeidelSmoother(A, b, x, postIters, tolPost, isDebug);

		return 0.0;
	}

	//!< Perform Multigrid using W-Cycle and Gauss-Seidel smoothing
	double  MultigridSolver::multiGridWCycleGS(Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& b, Eigen::MatrixXd& x, int k, bool isDebug)
	{
		double tolPre = 1e-15, tolPost = 1e-15;
		//[1] Smoothing
		GaussSeidelSmoother(A, b, x, preIters, tolPre, isDebug);

		//[2] Residual
		Eigen::MatrixXd res = b - A * x;

		//[3] Restriction
		Eigen::MatrixXd resRest = U[k].transpose() * res;

		//[4] Recursion
		Eigen::MatrixXd eps;
		eps.setZero(resRest.rows(), resRest.cols());
		if (k == U.size() - 1) {
			eps = coarsestSolver.solve(resRest);
		} else {
			multiGridWCycleGS(Abar[k + 1], resRest, eps, k + 1, isDebug);
		}

		//[5] Prolongation
		x = x + U[k] * eps;

		//[6] Post-Smoothing
		GaussSeidelSmoother(A, b, x, postIters, tolPost, isDebug);

		//[7] Residual
		res = b - A * x;

		//[8] Restriction
		resRest = U[k].transpose() * res;

		//[9] Recursion
		if (k == DoF.size() - 2) {
			eps = coarsestSolver.solve(resRest);
		} else {
			multiGridWCycleGS(Abar[k + 1], resRest, eps, k + 1, isDebug);
		}

		//[10] Prolongation
		x = x + U[k] * eps;

		//[11] Post-Smoothing
		GaussSeidelSmoother(A, b, x, postIters, tolPost, isDebug);

		return 0.0;
	}

	void MultigridSolver::GaussSeidelSmoother(Eigen::SparseMatrix<double>& LHS, Eigen::MatrixXd& rhs,
		Eigen::MatrixXd& x, int maxIter, double tol, bool isDebug)
	{
		int dim = x.cols();
		if (dim == 1) {
			for (int i = 0; i < maxIter; i++) {
				for (int k = 0; k < LHS.outerSize(); ++k) {
					double sum = 0.0;
					for (Eigen::SparseMatrix<double>::InnerIterator it(LHS, k); it; ++it) {
						if (it.row() != k) {
							sum += it.value() * x(it.row());
						}
					}
					x(k) = (rhs(k) - sum) / LHS.coeffRef(k, k);
				}
			}
		} else {
			for (int i = 0; i < maxIter; i++) {
				for (int iCol = 0; iCol < dim; ++iCol) {
					for (int k = 0; k < LHS.outerSize(); ++k) {
						double sum = 0.0;
						for (Eigen::SparseMatrix<double>::InnerIterator it(LHS, k); it; ++it) {
							if (it.row() != k) {
								sum += it.value() * x(it.row(), iCol);
							}
						}
						x(k, iCol) = (rhs(k, iCol) - sum) / LHS.coeffRef(k, k);
					}
				}
			}
		}
	}

	double MultigridSolver::residualCheck(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& b, const Eigen::MatrixXd& x, int type)
	{
		// 0: rel. norm (||Ax-b||/||b||)		1: L2 M^-1 (||Ax-b||_{M-1}/||b||_{M-1})		2: L2 M (||Ax-b||{M}/||b||_{M})		3: Abs (Ax-b).norm()
		double residue;
		switch (type)
		{
		case 0:
		{
			Eigen::VectorXd absResidue(b.cols());
			for (int i = 0; i < b.cols(); ++i) {
				absResidue(i) = (A * x.col(i) - b.col(i)).norm() / b.col(i).norm();
			}
			residue = absResidue.maxCoeff();
			break;
		}
		case 1:
		{
			Eigen::VectorXd absResidue(b.cols());
			Eigen::VectorXd residual;
			for (int i = 0; i < b.cols(); ++i) {
				residual = A * x.col(i) - b.col(i);
				double n1 = residual.transpose() * Minv * residual;
				double n2 = b.col(i).transpose() * Minv * b.col(i);
				absResidue(i) = sqrt(n1 / n2);
			}
			residue = absResidue.maxCoeff();
			break;
		}
		case 2:
		{
			Eigen::VectorXd absResidue(b.cols());
			Eigen::VectorXd residual;
			for (int i = 0; i < b.cols(); ++i) {
				residual = A * x.col(i) - b.col(i);
				double n1 = residual.transpose() * M * residual;
				double n2 = b.col(i).transpose() * M * b.col(i);
				absResidue(i) = sqrt(n1 / n2);
			}
			residue = absResidue.maxCoeff();
			break;
		}
		case 3:
			residue = (A * x - b).norm();
			break;
		default:
			break;
		}

		return residue;
	}

	void MultigridSolver::solve(Eigen::SparseMatrix<double>& LHS, Eigen::MatrixXd& rhs, Eigen::MatrixXd& x)
	{
		std::chrono::high_resolution_clock::time_point t0, t1, t2;
		std::chrono::duration<double> duration1, duration2;

		Eigen::MatrixXd mx = rhs;

		double residue = std::numeric_limits<double>::max();
		Eigen::VectorXd rhs0;
		Eigen::VectorXd rhs1;
		Eigen::VectorXd rhs2;
		rhs0 = mx.col(0);
		if (mx.cols() > 1) {
			rhs1 = mx.col(1);
			rhs2 = mx.col(2);
		}

		if (isSmootherGaussSeidel)
		{

			Abar.resize(U.size() + 1);
			//Mbar.resize(DoF.size());
			Abar[1] = U[0].transpose() * LHS * U[0];
			for (int k = 2; k < U.size() + 1; ++k) {
				Abar[k] = U[k - 1].transpose() * Abar[k - 1] * U[k - 1];
			}

			coarsestSolver.compute(Abar[U.size()]);

			int iterCount = 0;
			if (cycleType == 0) {
				Eigen::Vector3d relResidue;
				do {
					residue = multiGridVCycleGS(LHS, mx, x, 0, false);
					residue = residualCheck(LHS, mx, x, stoppingCriteria);
					++iterCount;
				} while ((residue > accuracy) && (iterCount < maxIter));
			} else if (cycleType == 1) {
				Eigen::Vector3d relResidue;
				do {
					residue = multiGridFCycleGS(LHS, mx, x, 0, false);
					residue = residualCheck(LHS, mx, x, stoppingCriteria);
					++iterCount;
				} while ((residue > accuracy) && (iterCount < maxIter));
			} else if (cycleType == 2) {
				Eigen::Vector3d relResidue;
				do {
					residue = multiGridWCycleGS(LHS, mx, x, 0, false);
					residue = residualCheck(LHS, mx, x, stoppingCriteria);
					++iterCount;
				} while ((residue > accuracy) && (iterCount < maxIter));
			} else {
				return;
			}

		}

	}

}

