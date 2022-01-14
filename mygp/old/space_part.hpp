#pragma once
#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>
#include <memory>

namespace gp {
	template <typename T>
	concept hasNorm = requires(T t) {
			t.norm();
		};

	template<typename T>
	class KdTree {
		std::vector<std::pair<uint32_t, T>> points;
		int depth;
		int dim;
		int median;
		std::shared_ptr<KdTree<T>> left, right;
		T min_corner, max_corner;

		std::vector<std::vector<std::pair<uint32_t, T>>> getChildPoints(std::shared_ptr<KdTree<T>>& tree) {
			if (tree->left == nullptr) return { tree->points };

			std::vector<std::vector<std::pair<uint32_t, T>>> p;
			std::vector<std::vector<std::pair<uint32_t, T>>> lp = getChildPoints(tree->left);
			std::vector<std::vector<std::pair<uint32_t, T>>> rp = getChildPoints(tree->right);

			p.insert(p.end(), lp.begin(), lp.end());
			p.insert(p.end(), rp.begin(), rp.end());

			return p;
		}

		bool boxIntersectsSphere(T min_corner, T max_corner, T center, float r) {
			if constexpr (hasNorm<T>) {
				float r2 = r;
				float dmin = (min_corner.array() - center.array()).max(0).pow(2).sum() +
                     (center.array() - max_corner.array()).max(0).pow(2).sum();
				return dmin <= r2;	

			} else {
				float r2 = r * r;
				float dmin = 0;
				for (size_t i = 0; i < dim; i++) {
					if (center[i] < min_corner[i]) dmin += pow(center[i] - min_corner[i], 2);
					else if (center[i] > max_corner[i]) dmin += pow(center[i] - max_corner[i], 2);
				}
				return dmin <= r2;
			}
		}

		double distance(const T a, const T b) {
			if constexpr (hasNorm<T>) {
				return (a - b).squaredNorm();
			} else {
				double s = 0.0;
				for (size_t i = 0; i < dim; i++) { s += pow(a[i] - b[i], 2); }
				return sqrt(s);
			}
		}

	public:
		KdTree(std::vector<T> points, int depth = 0, int dim = 3) : points(points.size()), depth(depth), dim(dim), median(0),
			left(nullptr), right(nullptr) {
			for (size_t i = 0; i < points.size(); i++) {
				this->points[i].first = i;
				this->points[i].second = points[i];
			}

			min_corner = points[0];
			max_corner = points[0];
			for (auto& p : points) {
				for (size_t i = 0; i < dim; i++) {
					std::tie(min_corner[i], max_corner[i])
						= std::minmax({ p[i], min_corner[i], max_corner[i] });
				}
			}
		}

		KdTree(std::vector<std::pair<uint32_t, T>> points, int depth = 0, int dim = 3) : points(points), depth(depth), dim(dim), median(0),
			left(nullptr), right(nullptr) {
			min_corner = points[0].second;
			max_corner = points[0].second;
			for (auto& p : points) {
				for (size_t i = 0; i < dim; i++) {
					std::tie(min_corner[i], max_corner[i])
						= std::minmax({ p.second[i], min_corner[i], max_corner[i] });
				}
			}
		}

		void subdiv() {
			if (left == nullptr) {
				int axis = this->depth % this->dim;
				
				auto compare = [axis](auto& l, auto& r) { return l.second[axis] < r.second[axis]; };
				std::sort(points.begin(), points.end(), compare);
				median = points.size() / 2;
				//median = std::distance(points.begin(), std::upper_bound(points.begin(), points.end(), points[median], compare));

				left = std::make_shared<KdTree>(std::vector<std::pair<uint32_t, T>>(points.begin(), points.begin() + median), depth + 1, dim);
				right = std::make_shared<KdTree>(std::vector<std::pair<uint32_t, T>>(points.begin() + median, points.end()), depth + 1, dim);

				return;
			}
			left->subdiv();
			right->subdiv();
		}

		std::vector<std::vector<std::pair<uint32_t, T>>> getPoints() {
			if (left == nullptr) return { points };

			std::vector<std::vector<std::pair<uint32_t, T>>> p;
			std::vector<std::vector<std::pair<uint32_t, T>>> lp = getChildPoints(left);
			std::vector<std::vector<std::pair<uint32_t, T>>> rp = getChildPoints(right);

			p.insert(p.end(), lp.begin(), lp.end());
			p.insert(p.end(), rp.begin(), rp.end());

			return p;
		}

		std::vector<std::pair<uint32_t, T>> getNearestRegion(T p) {
			if (left == nullptr) return points;

			int axis = this->depth % this->dim;
			auto tree = points[this->median].second[axis] > p[axis] ? left : right;
			while (tree->left != nullptr) {
				axis = tree->depth % this->dim;
				tree = tree->points[tree->median].second[axis] > p[axis] ? tree->left : tree->right;
			}
			
			return tree->points;
		}

		void getNearestNeighbor(const std::shared_ptr<KdTree<T>>& tree, const T p, 
																							double& min_dist, uint32_t& nearest_p) {
			if (tree->left == nullptr) {
				for (auto& t : tree->points) {
					double dist = distance(t.second, p);
					if (dist < min_dist) {
						min_dist = dist;
						nearest_p = t.first;
					}
				}
				return;
			} 

			int axis = tree->depth % tree->dim;
			if (tree->points[tree->median].second[axis] > p[axis]) {
				getNearestNeighbor(tree->left, p, min_dist, nearest_p);
				if (boxIntersectsSphere(tree->right->min_corner, tree->right->max_corner, p, min_dist))
					getNearestNeighbor(tree->right, p, min_dist, nearest_p);
			} else {
				getNearestNeighbor(tree->right, p, min_dist, nearest_p);
				if (boxIntersectsSphere(tree->left->min_corner, tree->left->max_corner, p, min_dist)) 
					getNearestNeighbor(tree->left, p, min_dist, nearest_p);
			}
		}

		uint32_t getNearestNeighbor(const T p) {
			uint32_t nearest_p = 0;
			double min_dist = std::numeric_limits<double>::max();

			if (left == nullptr) { return nearest_p; }

			int axis = this->depth % this->dim;
			auto tree = points[this->median].second[axis] > p[axis] ? left : right;

			if (points[this->median].second[axis] > p[axis]) {
				getNearestNeighbor(left, p, min_dist, nearest_p);
				if (boxIntersectsSphere(right->min_corner, right->max_corner, p, min_dist))
					getNearestNeighbor(right, p, min_dist, nearest_p);
				
			} else {
				getNearestNeighbor(right, p, min_dist, nearest_p);
				if (boxIntersectsSphere(left->min_corner, left->max_corner, p, min_dist)) 
					getNearestNeighbor(left, p, min_dist, nearest_p);
			}

			return nearest_p;
		}

	};


}