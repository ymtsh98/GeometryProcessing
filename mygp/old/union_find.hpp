#pragma once
#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>

namespace gp {
	class UnionFind {
		size_t numNode;
		std::vector<size_t> parents;

	public:
		UnionFind(const size_t n) : numNode(n), parents(n) {
			std::iota(parents.begin(), parents.end(), 0);
		}

		size_t root(const size_t node) {
			if (node == parents[node]) {
				return node;
			} else {
				return parents[node] = root(parents[node]);
			}
		}

		bool unite(const size_t lnode, const size_t rnode) {
			size_t lr = root(lnode), rr = root(rnode);
			if (lr == rr) return false;

			parents[lr] = rr;
			return true;
		}

		bool same(const size_t lnode, const size_t rnode) {
			return root(lnode) == root(rnode);
		}
	};
}