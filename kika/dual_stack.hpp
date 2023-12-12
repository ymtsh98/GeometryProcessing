#pragma once

#include <vector>
#include <tuple>
#include <utility>

namespace kika::util {
	template <typename TupleType, size_t AccessIndex = 0, typename RangeType = std::tuple_element_t<AccessIndex, TupleType>>
	class DualStack {
		std::vector<TupleType> laika, felicette;
		RangeType rangeStep, rangeAccum;

		void update() {
			laika.swap(felicette);
			felicette.clear();
			rangeAccum += rangeStep;
		}

	public:
		DualStack(const RangeType& rangeStep) : laika({}), felicette({}), rangeStep(rangeStep), rangeAccum(rangeStep) {}

		void push(const TupleType& value) {
			if (std::get<AccessIndex>(value) < rangeAccum) laika.push_back(value);
			else felicette.push_back(value);
		}

		void push(TupleType&& value) {
			if (std::get<AccessIndex>(value) < rangeAccum) laika.push_back(std::move(value));
			else felicette.push_back(std::move(value));
		}

		template <class... Args>
		void emplace(Args&&... vals) {
			if (std::get<AccessIndex>(std::forward_as_tuple(std::forward<Args>(vals)...)) < rangeAccum)
				laika.emplace_back(std::forward<Args>(vals)...);
			else felicette.emplace_back(std::forward<Args>(vals)...);
		}

		size_t size() const {
			return laika.size() + felicette.size();
		}

		bool empty() const {
			return laika.empty() && felicette.empty();
		}

		const TupleType& top() {
			if (laika.size() == 0) update();
			return laika.back();
		}

		void pop() {
			laika.pop_back();
			if (laika.size() == 0) update();
		}

		void swap(DualStack& dstack) {
			std::swap(rangeStep, dstack.rangeStep);
			std::swap(rangeAccum, dstack.rangeAccum);
			laika.swap(dstack.laika);
			felicette.swap(dstack.felicette);
		}

		void reserve(const size_t size) {
			laika.reserve(size);
			felicette.reserve(size);
		}
	};
}