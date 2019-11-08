#pragma once

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <opencv/cv.hpp>

class Vec3
{
public:
	double e[3];

	Vec3() { e[0] = 0; e[1] = 0; e[2] = 0; }
	Vec3(double x, double y, double z) { e[0] = x; e[1] = y; e[2] = z; }

	inline double x() const { return e[0]; }
	inline double y() const { return e[1]; }
	inline double z() const { return e[2]; }
	inline double r() const { return e[0]; }
	inline double g() const { return e[1]; }
	inline double b() const { return e[2]; }

	inline cv::Vec3b toCVPix() const;

	inline const Vec3& operator+() const { return *this; }
	inline Vec3	operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
	inline double operator[](int i) const { return e[i]; }
	inline double& operator[](int i) { return e[i]; }

	inline Vec3& operator+=(const Vec3& v2);
	inline Vec3& operator-=(const Vec3& v2);
	inline Vec3& operator*=(const Vec3& v2);
	inline Vec3& operator/=(const Vec3& v2);
	inline Vec3& operator*=(const double t);
	inline Vec3& operator/=(const double t);

	friend inline std::istream& operator>>(std::istream& is, Vec3& t);
	friend inline std::ostream& operator<<(std::ostream& os, Vec3& t);

	inline double length() const;
	inline double squared_length() const;
	inline void makeUnitVector();
};