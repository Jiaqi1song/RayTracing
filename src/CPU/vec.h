#ifndef VEC_H
#define VEC_H

#include <random>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>

/**
 * Define vec3 and point3.
 * Various operations are supported.
    - negate for both.
    - equal / not equal for both.
    - vec3 length / length_squared.
    - vec3 unit_vector, random_unit_vector, random_in_unit_disk, random_on_hemishpere
    - vec3 + vec3: vec3. Adds two vectors component-wise.
    - vec3 - vec3: vec3. Subtracts two vectors component-wise.
    - vec3 * double: vec3. Scales the vector by a scalar.
    - vec3 / double: vec3. Scales the vector inversely by a scalar.
    - vec3 * vec3 : double. Dot product between two vectors.
    - vec3 x vec3: vec3. Cross product between two vextors.
    - vec3 reflect and refract.
    - point3 distance to origin / between points.
    - point3 + vec3: point3. Translates the point by a vector.
    - point3 - vec3: point3. Translates the point in the opposite direction of the vector.
    - point3 - point3: vec3. Computes the displacement vector between two points.
 * Part of operations are inplace supported.
 */

/* ********** Utils ********** */

// Some constants defined here.
constexpr double EPSILON = 1e-8;             ///< Tolerance value used for floating point comparisons.
constexpr double PI = 3.1415926535897932385; ///< Constant value for PI.

/**
 * Generate a random double in [0, 1).
 *
 * @return A random double in the range [0, 1).
 */
inline double random_double()
{
    // Returns a random real in [0,1).
    return std::rand() / (RAND_MAX + 1.0);
}

/**
 * Generate a random double in [min, max).
 *
 * @param min Minimum value.
 * @param max Maximum value.
 * @return A random double in the range [min, max).
 */
inline double random_double(double min, double max)
{
    // Returns a random real in [min,max).
    return min + (max - min) * random_double();
}

/**
 * Convert degrees to radians.
 *
 * @param degrees Angle in degrees.
 * @return Equivalent angle in radians.
 */
inline double degrees_to_radians(double degrees)
{
    return degrees * PI / 180.0;
}

/* ********** vec_base ********** */

/**
 * Base class for 3D vectors and points.
 * Provides storage and indexing functionality.
 */
class vec_base
{
protected:
    // Data is accessed via vec_base[i]
    double e[3];

public:
    /**
     * Default constructor that initializes all components to 0.
     */
    vec_base() : e{0.0, 0.0, 0.0} {}

    /**
     * Constructor that initializes components to given values.
     *
     * @param e0 X component.
     * @param e1 Y component.
     * @param e2 Z component.
     */
    vec_base(double e0, double e1, double e2) : e{e0, e1, e2} {}

    /**
     * Access element at the given index (read-only).
     *
     * @param i Index (0-based).
     * @return The value at the index.
     * @throws std::out_of_range if index is invalid.
     */
    double operator[](int i) const
    {
        if (i < 0 || i > 2)
            throw std::out_of_range("Index out of bounds");
        return e[i];
    }

    /**
     * Access element at the given index (read-write).
     *
     * @param i Index (0-based).
     * @return Reference to the value at the index.
     * @throws std::out_of_range if index is invalid.
     */
    double &operator[](int i)
    {
        if (i < 0 || i > 2)
            throw std::out_of_range("Index out of bounds");
        return e[i];
    }
};

/* ********** vec3 ********** */

/**
 * Class representing a 3D vector.
 * Inherits from vec_base and provides additional vector operations.
 */
class vec3 : public vec_base
{
public:
    /**
     * Constructor to initialize the vector with given x, y, z values.
     *
     * @param x X component (default 0.0).
     * @param y Y component (default 0.0).
     * @param z Z component (default 0.0).
     */
    vec3(double x = 0.0, double y = 0.0, double z = 0.0) : vec_base(x, y, z) {}

    /**
     * Getter for x component.
     *
     * @return The x component of the vector.
     */
    double x() const { return e[0]; }

    /**
     * Getter for y component.
     *
     * @return The y component of the vector.
     */
    double y() const { return e[1]; }

    /**
     * Getter for z component.
     *
     * @return The z component of the vector.
     */
    double z() const { return e[2]; }

    /**
     * Negate the vector.
     *
     * @return A new vector that is the negation of this vector.
     */
    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

    /**
     * Add another vector to this vector (in-place).
     *
     * @param v Vector to add.
     * @return Reference to this vector after addition.
     */
    vec3 &operator+=(const vec3 &v)
    {
        e[0] += v[0];
        e[1] += v[1];
        e[2] += v[2];
        return *this;
    }

    /**
     * Subtract another vector from this vector (in-place).
     *
     * @param v Vector to subtract.
     * @return Reference to this vector after subtraction.
     */
    vec3 &operator-=(const vec3 &v)
    {
        e[0] -= v[0];
        e[1] -= v[1];
        e[2] -= v[2];
        return *this;
    }

    /**
     * Scale this vector by a scalar (in-place).
     *
     * @param t Scalar multiplier.
     * @return Reference to this vector after scaling.
     */
    vec3 &operator*=(double t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    /**
     * Divide the vector by a scalar (in-place).
     *
     * @param t Scalar divisor.
     * @return Reference to the modified vector.
     * @throws std::runtime_error if t is zero.
     */
    vec3 &operator/=(double t)
    {
        if (t == 0)
            throw std::runtime_error("Division by zero");
        double inv_t = 1.0 / t;
        e[0] *= inv_t;
        e[1] *= inv_t;
        e[2] *= inv_t;
        return *this;
    }

    /**
     * Compute the squared length of the vector.
     *
     * @return The squared length of the vector.
     */
    double length_squared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    /**
     * Compute the length of the vector.
     *
     * @return The length of the vector.
     */
    double length() const
    {
        return std::sqrt(length_squared());
    }

    /**
     * Generate a random vector with components in the range [0, 1).
     *
     * @return A random vector.
     */
    static vec3 random()
    {
        static std::random_device rd;                          // Seed for random number engine
        static std::mt19937 gen(rd());                         // Mersenne Twister random number generator
        static std::uniform_real_distribution<> dis(0.0, 1.0); // Distribution in [0, 1)

        return vec3(dis(gen), dis(gen), dis(gen));
    }

    /**
     * Generate a random vector with components in the range [min, max).
     *
     * @param min Minimum value for components.
     * @param max Maximum value for components.
     * @return A random vector.
     */
    static vec3 random(double min, double max)
    {
        static std::random_device rd;                          // Seed for random number engine
        static std::mt19937 gen(rd());                         // Mersenne Twister random number generator
        static std::uniform_real_distribution<> dis(0.0, 1.0); // Distribution in [0, 1)

        double rand_x = min + (max - min) * dis(gen);
        double rand_y = min + (max - min) * dis(gen);
        double rand_z = min + (max - min) * dis(gen);
        return vec3(rand_x, rand_y, rand_z);
    }

    /**
     * Check if the vector is approximately zero in all components.
     * This actually judges if the vector has zero length.
     *
     * @return True if all components are near zero, false otherwise.
     */
    bool near_zero() const
    {
        return (std::fabs(e[0]) < EPSILON && std::fabs(e[1]) < EPSILON && std::fabs(e[2]) < EPSILON);
    }
};

/**
 * Output stream operator for vec3.
 *
 * @param out Output stream.
 * @param v Vector to output.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &out, const vec3 &v)
{
    return out << "vec3(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
}

/**
 * Equality operator for vec3.
 *
 * @param u First vector.
 * @param v Second vector.
 * @return True if the vectors are equal, false otherwise.
 */
inline bool operator==(const vec3 &u, const vec3 &v)
{
    return std::fabs(u[0] - v[0]) < EPSILON &&
           std::fabs(u[1] - v[1]) < EPSILON &&
           std::fabs(u[2] - v[2]) < EPSILON;
}

/**
 * Inequality operator for vec3.
 *
 * @param u First vector.
 * @param v Second vector.
 * @return True if the vectors are not equal, false otherwise.
 */
inline bool operator!=(const vec3 &u, const vec3 &v)
{
    return !(u == v);
}

/**
 * Vector addition.
 *
 * @param u First vector.
 * @param v Second vector.
 * @return The result of adding the two vectors.
 */
inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}

/**
 * Vector subtraction.
 *
 * @param u First vector.
 * @param v Second vector.
 * @return The result of subtracting the second vector from the first.
 */
inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

/**
 * Scalar multiplication.
 *
 * @param u Vector.
 * @param t Scalar.
 * @return The result of scaling the vector by the scalar.
 */
inline vec3 operator*(const vec3 &u, double t)
{
    return vec3(u[0] * t, u[1] * t, u[2] * t);
}

/**
 * Scalar multiplication (commutative).
 *
 * @param t Scalar.
 * @param v Vector.
 * @return The result of scaling the vector by the scalar.
 */
inline vec3 operator*(double t, const vec3 &v)
{
    return vec3(t * v[0], t * v[1], t * v[2]);
}

/**
 * Scalar division.
 *
 * @param u Vector.
 * @param t Scalar.
 * @return The result of dividing the vector by the scalar.
 * @throws std::runtime_error if t is zero.
 */
inline vec3 operator/(const vec3 &u, double t)
{
    if (t == 0)
        throw std::runtime_error("Division by zero");
    double inv_t = 1.0 / t;
    return vec3(u[0] * inv_t, u[1] * inv_t, u[2] * inv_t);
}

/**
 * Dot product of two vectors.
 *
 * @param u First vector.
 * @param v Second vector.
 * @return The dot product of the two vectors.
 */
inline double dot(const vec3 &u, const vec3 &v)
{
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

/**
 * Cross product of two vectors.
 *
 * @param u First vector.
 * @param v Second vector.
 * @return The cross product of the two vectors.
 */
inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]);
}

/**
 * Compute the unit vector of the given vector.
 *
 * @param v Vector to normalize.
 * @return The unit vector.
 */
inline vec3 unit_vector(const vec3 &v)
{
    return v.near_zero() ? vec3() : v / v.length();
}

/**
 * Generate a random unit vector.
 *
 * @return A random unit vector.
 *
 * The original method is not very efficient
 * because the probability of selecting a valid vector
 * within the sphere from a cube is about 52%.
 * Since the vector is always made sure to have length 1,
 * There is no need to check if the vector is too small.
 *
 * Old code

// inline vec3 random_unit_vector()
// {
//     while (true)
//     {
//         vec3 v = vec3::random(-1, 1);
//         double len_sq = v.length_squared();
//         if (1e-160 < len_sq && len_sq <= 1)
//             return v / std::sqrt(len_sq);
//     }
// }

 */
inline vec3 random_unit_vector()
{
    double z = random_double(-1, 1);       // Sample z in [-1, 1]
    double phi = random_double(0, 2 * PI); // Sample azimuth angle in [0, 2π]
    double r = std::sqrt(1 - z * z);       // Radius in xy-plane
    return vec3(r * std::cos(phi), r * std::sin(phi), z);
}

/**
 * Generate a random vector within the unit disk in the xy-plane.
 *
 * @return A random vector in the unit disk.
 *
 * Old code

// inline vec3 random_in_unit_disk()
// {
//     while (true)
//     {
//         vec3 vec = vec3(random_double(-1, 1), random_double(-1, 1), 0);
//         if (vec.length_squared() < 1)
//             return vec;
//     }
// }

 */
inline vec3 random_in_unit_disk()
{
    double r = std::sqrt(random_double());
    double theta = random_double(0, 2 * PI);

    double x = r * std::cos(theta);
    double y = r * std::sin(theta);

    return vec3(x, y, 0);
}

/**
 * Generate a random vector on a unit sphere.
 *
 * @return A random vector on the sphere.
 */
inline vec3 random_on_hemisphere(const vec3 &normal)
{
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

/**
 * Reflect a vector about a normal.
 *
 * @param v Incident vector.
 * @param n Normal vector.
 * @return The reflected vector.
 */
inline vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - 2 * dot(v, n) * n;
}

/**
 * Refract a vector through a surface with a given normal.
 *
 * @param v Incident vector.
 * @param n Normal vector.
 * @param eta Relative index of refraction.
 * @return The refracted vector.
 */
inline vec3 refract(const vec3 &v, const vec3 &n, double eta)
{
    double cos_theta = std::fmin(dot(-v, n), 1.0);
    vec3 r_out_perp = eta * (v + cos_theta * n);
    vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

/* ********** point3 ********** */

/**
 * Class representing a 3D point.
 * Inherits from vec_base and provides point-specific operations.
 */
class point3 : public vec_base
{
public:
    /**
     * Constructor to initialize the point with given x, y, z values.
     *
     * @param x X component (default 0.0).
     * @param y Y component (default 0.0).
     * @param z Z component (default 0.0).
     */
    point3(double x = 0.0, double y = 0.0, double z = 0.0) : vec_base(x, y, z) {}

    /**
     * Getter for x component.
     *
     * @return The x component of the point.
     */
    double x() const { return e[0]; }

    /**
     * Getter for y component.
     *
     * @return The y component of the point.
     */
    double y() const { return e[1]; }

    /**
     * Getter for z component.
     *
     * @return The z component of the point.
     */
    double z() const { return e[2]; }

    /**
     * Negate the point (invert all components).
     *
     * @return A new point that is the negation of this point.
     */
    point3 operator-() const { return point3(-e[0], -e[1], -e[2]); }

    /**
     * Add another vector from this vector (in-place).
     *
     * @param v Vector to add.
     * @return Reference to this point after translation.
     */
    point3 &operator+=(const vec3 &v)
    {
        e[0] += v[0];
        e[1] += v[1];
        e[2] += v[2];
        return *this;
    }

    /**
     * Subtract another vector from this vector (in-place).
     *
     * @param v Vector to subtract.
     * @return Reference to this point after translation.
     */
    point3 &operator-=(const vec3 &v)
    {
        e[0] -= v[0];
        e[1] -= v[1];
        e[2] -= v[2];
        return *this;
    }
};

/**
 * Output stream operator for point3.
 *
 * @param out Output stream.
 * @param p Point to output.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &out, const point3 &p)
{
    return out << "point3(" << p.x() << ", " << p.y() << ", " << p.z() << ")";
}

/**
 * Compute the distance from the point to the origin.
 *
 * @param p Point to compute distance from.
 * @return The distance to the origin.
 */
inline double distance_to_origin(const point3 &p)
{
    return std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
}

/**
 * Compute the distance between two points.
 *
 * @param p1 First point.
 * @param p2 Second point.
 * @return The distance between the two points.
 */
inline double distance(const point3 &p1, const point3 &p2)
{
    return std::sqrt(
        (p1.x() - p2.x()) * (p1.x() - p2.x()) +
        (p1.y() - p2.y()) * (p1.y() - p2.y()) +
        (p1.z() - p2.z()) * (p1.z() - p2.z()));
}

/**
 * Equality operator for point3.
 *
 * @param u First point.
 * @param v Second point.
 * @return True if the points are equal, false otherwise.
 */
inline bool operator==(const point3 &u, const point3 &v)
{
    return std::fabs(u[0] - v[0]) < EPSILON &&
           std::fabs(u[1] - v[1]) < EPSILON &&
           std::fabs(u[2] - v[2]) < EPSILON;
}

/**
 * Inequality operator for point3.
 *
 * @param u First point.
 * @param v Second point.
 * @return True if the points are not equal, false otherwise.
 */
inline bool operator!=(const point3 &u, const point3 &v)
{
    return !(u == v);
}

/**
 * Subtract one point from another to get a displacement vector.
 * point3 - point3 = vec3
 *
 * @param p1 First point.
 * @param p2 Second point.
 * @return The displacement vector from p2 to p1.
 */
inline vec3 operator-(point3 p1, point3 p2)
{
    return vec3(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
}

/**
 * Add a vector to a point.
 * point3 + vec3 = point3
 *
 * @param p Point to add to.
 * @param v Vector to add.
 * @return The resulting point.
 */
inline point3 operator+(point3 p, vec3 v)
{
    return point3(p[0] + v[0], p[1] + v[1], p[2] + v[2]);
}

/**
 * Add a vector to a point (commutative).
 * vec3 + point3 = point3
 *
 * @param v Vector to add.
 * @param p Point to add to.
 * @return The resulting point.
 */
inline point3 operator+(vec3 v, point3 p)
{
    return point3(p[0] + v[0], p[1] + v[1], p[2] + v[2]);
}

/**
 * Subtract a vector from a point.
 * point3 - vec3 = point3
 *
 * @param p Point to subtract from.
 * @param v Vector to subtract.
 * @return The resulting point.
 */
inline point3 operator-(point3 p, vec3 v)
{
    return point3(p[0] - v[0], p[1] - v[1], p[2] - v[2]);
}

#endif
