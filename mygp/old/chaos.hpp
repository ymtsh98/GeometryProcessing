#pragma once
#include <functional>
#include <numbers>
#include <random>
#include <utility>
#include <vector>
#include <glm/glm.hpp>

namespace gp {
	class ChaosMap {
	public:
		std::pair<float, float> range;
		std::pair<int, int> iter;
		int period;
		int time;

		std::pair<float, float> offset;
		std::pair<float, float> scale;

		std::vector<double> param;
		std::vector<std::pair<double, double>> param_range;
		std::vector<int> update_param_idx;

		ChaosMap(const std::pair<float, float>& range, const std::pair<int, int>& iter, const int& period, 
			const int& time, const std::pair<float, float>& offset, const std::pair<float, float>& scale, 
			const std::vector<double>& param, const std::vector<std::pair<double, double>>& param_range, const std::vector<int>& update_param_idx) : 
			range(range), iter(iter), period(period), time(time), offset(offset), scale(scale), 
			param(param), param_range(param_range), update_param_idx(update_param_idx) {}

		virtual void generate(const int& i, double& x, double& y, double& z) { x = 0; y = 0; z = 0; }
		virtual void function(double& x, double& y, double& z) {}

		std::vector<glm::vec3> getPoints() {
			std::vector<glm::vec3> points;
			points.reserve(iter.first * iter.second);

			for (int i = 0; i < iter.first; i++) {
				double x, y, z;
				generate(i, x, y, z);

				for (int j = 0; j < iter.second; j++) {
					function(x, y, z);

					if (j > period)
						points.emplace_back(x * scale.first + offset.first,
							y * scale.second + offset.second, z * scale.first);
				}
			}

			return points;
		}

		void update() {
			time++;
			for (auto& i : update_param_idx) {
				param[i] = (sin(std::numbers::pi / 180.0 * (time % 360)) + 1.0) / 2.0 *
					(param_range[i].second - param_range[i].first) + param_range[i].first;
			}
		}

		void update(std::vector<int> param_idx) {
			time++;
			for (auto& i : param_idx) {
				param[i] = (sin(std::numbers::pi / 180.0 * (time % 360)) + 1.0) / 2.0 *
					(param_range[i].second - param_range[i].first) + param_range[i].first;
			}
		}

	};

	class LogisticMap : public ChaosMap{
	public:
		LogisticMap() : ChaosMap({ 0.0, 4.0 }, { 1e+3, 1e+3 }, -1, 0, { 0.0, 0.0 }, { 1.0, 1.0 }, { 0.0 }, { {0.0, 1.0} }, { 0 }) {}

		double f(const double& x, const double& y, const double& z) { x * y * (1.0 - y); }

		void generate(const int& i, double& x, double& y, double& z) {
			x = (double)i / iter.first * (range.second - range.first) + range.first;
			y = param[0];
		}

		void function(double& x, double& y, double& z) {
			y = x * y * (1.0 - y);
		}
	};
		
	class IkedaMap : public ChaosMap {
	public:
		IkedaMap() : ChaosMap({ 0.0, 1.0 }, { 1e+3, 1e+3 }, -1, 0, { 0.0, 0.0 }, { 1.0, 1.0 }, { 0.5 }, { {0.0, 1.0 } }, { 0 }) {}

		void generate(const int& i, double& x, double& y, double& z) {
			x = (double)rand() / RAND_MAX * (range.second - range.first) + range.first;
			y = (double)rand() / RAND_MAX * (range.second - range.first) + range.first;
		}

		void function(double& x, double& y, double& z) {
			double t = 0.4 - 6.0 / (1 + x * x + y * y);
			auto tx = 1.0 + param[0] * (x * cos(t) - y * sin(t));
			y = param[0] * (x * sin(t) + y * cos(t));
			x = tx;
		}

	};

	class TinkerbellMap : public ChaosMap {
	public:

		TinkerbellMap() : ChaosMap({ -1.0, 0.5 }, { 1e+2, 1e+4 }, -1, 0, 
															 { 0.0, 0.0 }, { 1.0, 1.0 }, { 0.9, -0.6013, 2.0, 0.5 }, { {0.0, 1.0}, {-1.0, 0.0}, {1.0, 2.0}, { 0.3, 0.5} }, { 2 }) {}

		void generate(const int& i, double& x, double& y, double& z) {
			x = (double)rand() / RAND_MAX * (range.second - range.first) + range.first;
			y = (double)rand() / RAND_MAX * (range.second - range.first) + range.first;
		}

		void function(double& x, double& y, double& z) {
			auto tx = x * x - y * y + param[0] * x + param[1] * y;
			y = 2 * x * y + param[2] * x + param[3] * y;
			x = tx;
		}
	};

	class BogdanovMap : public ChaosMap {
		void generate(const int& i, double& x, double& y, double& z) {
			x = (double)rand() / RAND_MAX * (range.second - range.first) + range.first;
			y = (double)rand() / RAND_MAX * (range.second - range.first) + range.first;
		}

		void function(double& x, double& y, double& z) {
			y = y + param[0] * y + param[1] * x * (x - 1.0) + param[2] * x * y;
			x = x + y;
		}

	public:
		BogdanovMap() : ChaosMap({ 0.0, 1.0 }, { 1e+2, 1e+4 }, -1, 0, { 0.0, 0.0 }, { 1.0, 1.0 }, 
														 { 0, 1.2, 0 }, { {0.0, 1.0}, {0.0, 2.0}, {0.0, 1.0} }, { 1 }) {}

	};

	class ThomasCyclicallySymmetricAttractor : public ChaosMap {
		void generate(const int& i, double& x, double& y, double& z) { 
			x = (double)rand() / RAND_MAX * (range.second - range.first) + range.first;
			y = (double)rand() / RAND_MAX * (range.second - range.first) + range.first;
			z = (double)rand() / RAND_MAX * (range.second - range.first) + range.first;
		}

		void function(double& x, double& y, double& z) {
			auto tx = x + (sin(y) - param[0] * x) * param[1];
			y = y + (sin(z) - param[0] * y) * param[1];
			z = z + (sin(x) - param[0] * z) * param[1];
			x = tx;
		}

	public:

		ThomasCyclicallySymmetricAttractor() : 
			ChaosMap({ -1.0, 1.0 }, { 1e+2, 1e+4 }, -1, 0, { 0.0, 0.0 }, { 1.0, 1.0 },
							 { 0.209, 0.5 }, { {0.0, 0.35}, {0.0, 1.0} }, { 0 }) {}

	};

	class GingerbreadmanMap : public ChaosMap {
		void generate(const int& i, double& x, double& y, double& z) {
			x = (double)rand() / RAND_MAX * (range.second - range.first) + range.first;
			y = (double)rand() / RAND_MAX * (range.second - range.first) + range.first;
		}

		void function(double& x, double& y, double& z) {
			auto tx = 1.0 - param[0] * y + param[1] * abs(x);
			y = x;
			x = tx;
		}

	public:
		GingerbreadmanMap() : ChaosMap({ -10.0, 10.0 }, { 1e+2, 1e+4 }, -1, 0, { 0.0, 0.0 }, { 1.0, 1.0 },
																	 { 1.0, 0.5 }, { {0.0, 1.0}, {0.0, 1.0} }, { 1 }) {}

	};
}