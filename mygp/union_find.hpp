#pragma once
#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>

namespace gp::util {
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

		std::vector<Index> getParentTree() {
			return parents;
		}
	};

	template <typename T1, typename T2>
	auto connectedMesh(const T1& pos, const T2& faces) {
		UnionFind uf(pos.size());
		for (const auto& face : faces) {
			for (size_t i = 1; i < face.size(); i++) uf.unite(face[0], face[i]);
		}

		std::unordered_map<size_t, std::vector<size_t>> s;
		for (size_t i = 0; i < pos.size(); i++) {
			s[uf.root(i)].emplace_back(i);
		}

		return s;
	}
}
