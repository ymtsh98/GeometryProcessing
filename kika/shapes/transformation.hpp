#pragma once

namespace kika::shape {

// Transformation parameters.
template <typename Prec=double>
class Transformation {
	struct Vector {
		Prec x, y, z;
		Vector() : x(0), y(0), z(0) {}
	};

public:
	Vector translate, rotate, scale;

	Transformation() {
		
	}
};

}