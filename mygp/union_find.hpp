#pragma once
#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>

namespace gp {
	template <typename Index = size_t>
	class UnionFind {
		Index numNode;
		std::vector<Index> parents;

	public:
		UnionFind(const Index n) : numNode(n), parents(n) {
			std::iota(parents.begin(), parents.end(), 0);
		}

		Index root(const Index node) {
			if (node == parents[node]) {
				return node;
			} else {
				return parents[node] = root(parents[node]);
			}
		}

		bool unite(const Index lnode, const Index rnode) {
			Index lr = root(lnode), rr = root(rnode);
			if (lr == rr) return false;

			parents[lr] = rr;
			return true;
		}

		bool same(const Index lnode, const Index rnode) {
			return root(lnode) == root(rnode);
		}
	};
}
