#pragma once
/*
@article{nasikun2018fastspectrum,
title={Fast Approximation of Laplace-–Beltrami Eigenproblems},
author={Nasikun, Ahmad and Brandt, Christopher and Hildebrandt, Klaus},
journal={Computer Graphics Forum},
volume={37},
number={5},
year={2018},
publisher={Wiley}
}
*/

namespace gp::impl::samp {
	struct BoxStruct {
		int		id;
		double	distance;
		bool	operator> (const BoxStruct& ref) const { return distance > ref.distance; }
		bool	operator< (const BoxStruct& ref) const { return distance < ref.distance; }
	};

	void VtoVDist(const Eigen::VectorXd& V1, const Eigen::VectorXd& V2, double& dist) {
		Eigen::VectorXd	  v = V2 - V1;
		dist				= v.norm();
	}
}

namespace gp {
	void constructPoissonDiskSample(const Eigen::MatrixXd& V, const int& numSamples, const int& n, const double& avgEdgeLength, Eigen::VectorXi& Sample) {
		using namespace std;
		using namespace impl::samp;
		Eigen::Vector3d minV(V.col(0).minCoeff(), V.col(1).minCoeff(), V.col(2).minCoeff());
		Eigen::Vector3d maxV(V.col(0).maxCoeff(), V.col(1).maxCoeff(), V.col(2).maxCoeff());
		Eigen::Vector3d range, boxDist;
		Eigen::Vector3i nn;

		range = maxV - minV;
		double minRange = range.minCoeff();

		for (int i = 0; i < 3; i++) {
			boxDist(i) = minRange / (double)n;
			nn(i) = (int)ceil(range(i) / boxDist(i));
		}

		set<int>		locSample;
		const int		SamplePerBox = 10;
		double			radius = sqrt((double)V.rows() / (2.5*(double)numSamples)) * avgEdgeLength;

		priority_queue<BoxStruct, std::vector<BoxStruct>, std::greater<BoxStruct>>	UnvisitedBoxQueue;
		list<int>																	VisitedEmptyBoxList;

		list<int>							v1;
		vector<list<int>>					v2;
		vector<vector<list<int>>>			v3;
		vector<vector<vector<list<int>>>>	v4;

		v2.reserve(nn(2));	v3.reserve(nn(1));	v4.reserve(nn(0));
		for (int i = 0; i < nn(2); i++) { v2.push_back(v1); }
		for (int i = 0; i < nn(1); i++) { v3.push_back(v2); }
		for (int i = 0; i < nn(0); i++) { v4.push_back(v3); }

		for (int vId = 0; vId < V.rows(); vId++) {
			int xId, yId, zId;
			xId = (int)abs(floor((V(vId, 0) - minV(0)) / boxDist(0)));
			yId = (int)abs(floor((V(vId, 1) - minV(1)) / boxDist(1)));
			zId = (int)abs(floor((V(vId, 2) - minV(2)) / boxDist(2)));
			if (xId >(nn(0) - 1)) xId = nn(0) - 1;
			if (yId >(nn(1) - 1)) yId = nn(1) - 1;
			if (zId >(nn(2) - 1)) zId = nn(2) - 1;

			v4[xId][yId][zId].push_back(vId);
		}

		std::random_device rd;
		std::default_random_engine generator(rd());
		int	trimCounter = 0;

		int poolCounter = 0;
		for (int iIdx = 0; iIdx < nn(0); iIdx++) {
			for (int jIdx = 0; jIdx < nn(1); jIdx++) {
				for (int kIdx = 0; kIdx < nn(2); kIdx++) {
					Eigen::Vector3d		boxMin(minV(0) + iIdx*boxDist(0), minV(1) + jIdx*boxDist(1), minV(2) + kIdx*boxDist(2));

					int boxSize = v4[iIdx][jIdx][kIdx].size();
					if (boxSize == 0) {
						VisitedEmptyBoxList.push_back(iIdx*nn(2)*nn(1) + jIdx*nn(2) + kIdx);
					} else {	
						double randDist = (double)(rand() % 100) / (double)10;
						BoxStruct curBox{ iIdx*nn(2)*nn(1) + jIdx*nn(2) + kIdx, randDist };
						UnvisitedBoxQueue.push(curBox);

						if (boxSize > SamplePerBox) {
							set<int> vertexToDel;
							srand(time(NULL));
							do {
								std::uniform_int_distribution<long> distribution(0, boxSize);
								int vSel = distribution(generator);
								vertexToDel.insert(vertexToDel.end(), vSel);
							} while (vertexToDel.size() <= (boxSize - SamplePerBox));

							int l = 0;
							for (list<int>::iterator it = v4[iIdx][jIdx][kIdx].begin(); it != v4[iIdx][jIdx][kIdx].end(); ) {
								if (vertexToDel.find(l) != vertexToDel.end()) {
									it = v4[iIdx][jIdx][kIdx].erase(it);
								} else {
									++it;
								}
								l++;
							}
							trimCounter++;
						}
					}
					poolCounter += v4[iIdx][jIdx][kIdx].size();

				}
			}
		} 
		while (VisitedEmptyBoxList.size()<nn(0)*nn(1)*nn(2)) {
			BoxStruct box1 = UnvisitedBoxQueue.top();
			UnvisitedBoxQueue.pop();

			int b_xId, b_yId, b_zId;
			b_xId = round(box1.id / (nn(1)*nn(2)));
			b_yId = round((box1.id - b_xId*nn(1)*nn(2)) / nn(2));
			b_zId = box1.id % nn(2);
			if (v4[b_xId][b_yId][b_zId].size() < 1) continue;

			srand(time(NULL));
			int randV1 = rand() % v4[b_xId][b_yId][b_zId].size();
			int boxSample;
			int locCounter = 0;
			for (list<int>::iterator it = v4[b_xId][b_yId][b_zId].begin(); it != v4[b_xId][b_yId][b_zId].end(); ++it) {
				if (randV1 == locCounter) boxSample = *it;
				locCounter++;
			}
			locSample.insert(locSample.end(), boxSample);

			for (int i = b_xId - 1; i <= b_xId + 1; i++) {
				if (i < 0 || i >(nn(0) - 1)) continue;
				for (int j = b_yId - 1; j <= b_yId + 1; j++) {
					if (j < 0 || j >(nn(1) - 1)) continue;
					for (int k = b_zId - 1; k <= b_zId + 1; k++) {
						if (k < 0 || k >(nn(2) - 1)) continue;
						if (v4[i][j][k].size() > 0) {
							for (list<int>::iterator it = v4[i][j][k].begin(); it != v4[i][j][k].end(); ) {
								double distAB;
								VtoVDist(V.row(boxSample), V.row(*it), distAB);

								if (distAB <= radius) {
									it = v4[i][j][k].erase(it);
								} else {
									++it;
								}
							} 
							if (v4[i][j][k].size() < 1) {
								VisitedEmptyBoxList.push_back(i*nn(1)*nn(2) + j*nn(2) + k);
							} else	{
								if (i == b_xId && j == b_yId && k == b_zId) {
									double nRandDist = (double)(rand() % 100) / (double)10;

									BoxStruct curBox{ i*nn(1)*nn(2) + j*nn(2) + k, nRandDist };
									UnvisitedBoxQueue.push(curBox);
								}
							}
						}
					} 
				}
			} 
		}

		Sample.resize(locSample.size());
		int sId = 0;
		for (int samp : locSample) {
			Sample(sId++) = samp;
		}
	}
}