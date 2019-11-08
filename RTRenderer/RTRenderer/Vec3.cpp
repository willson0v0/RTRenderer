#include "Vec3.h"

inline cv::Vec3b Vec3::toCVPix() const
{
	return cv::Vec3b(
		uchar(e[2] * 255.99),
		uchar(e[1] * 255.99),
		uchar(e[0] * 255.99));
}

inline Vec3& Vec3::operator+=(const Vec3& v2)
{
	e[0] += v2.x();
	e[1] += v2.y();
	e[2] += v2.z();
	return *this;
}

inline Vec3& Vec3::operator-=(const Vec3& v2)
{
	e[0] -= v2.x();
	e[1] -= v2.y();
	e[2] -= v2.z();
	return *this;
}

inline Vec3& Vec3::operator*=(const Vec3& v2)
{
	e[0] *= v2.x();
	e[1] *= v2.y();
	e[2] *= v2.z();
	return *this;
}

inline Vec3& Vec3::operator/=(const Vec3& v2)
{
	e[0] /= v2.x();
	e[1] /= v2.y();
	e[2] /= v2.z();
	return *this;
}

inline Vec3& Vec3::operator*=(const double t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

inline Vec3& Vec3::operator/=(const double t)
{
	e[0] /= t;
	e[1] /= t;
	e[2] /= t;
	return *this;
}

inline double dot(const Vec3& v1, const Vec3& v2)
{
	return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
}

inline Vec3 cross(const Vec3& v1, const Vec3& v2)
{
	return Vec3(
		v1[1] * v2[2] - v1[2] * v2[1],
		v1[2] * v2[0] - v1[0] * v2[2],
		v1[0] * v2[1] - v1[1] * v2[0]);
}

inline Vec3 operator+(const Vec3& v1, const Vec3& v2)
{
	return Vec3(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
}

inline Vec3 operator-(const Vec3& v1, const Vec3& v2)
{
	return Vec3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
}

inline Vec3 operator*(const Vec3& v1, const Vec3& v2)
{
	return Vec3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
}

inline Vec3 operator/(const Vec3& v1, const Vec3& v2)
{
	return Vec3(v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2]);
}

inline Vec3 operator*(const Vec3& v, double t)
{
	return Vec3(v[0] * t, v[1] * t, v[2] * t);
}

inline Vec3 operator*(double t, const Vec3& v)
{
	return v * t;
}

inline Vec3 operator/(const Vec3& v, double t)
{
	return Vec3(v[0] / t, v[1] / t, v[2] / t);
}

inline double Vec3::length() const
{
	return sqrt(this->squared_length());
}

inline double Vec3::squared_length() const
{
	return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}

inline void Vec3::makeUnitVector()
{
	double k = 1.0 / this->length();
	(*this) *= k;
}

inline std::istream& operator>>(std::istream& is, Vec3& t)
{
	is >> t.e[0] >> t.e[1] >> t.e[2];
	return is;
}

inline std::ostream& operator<<(std::ostream& os, Vec3& t)
{
	os << t.e[0] << " " << t.e[1] << " " << t.e[2];
	return os;
}

inline Vec3 unitVector(Vec3 v)
{
	return v / v.length();
}