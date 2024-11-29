#ifndef RAY_H
#define RAY_H

#include "vec.h"

/**
 * Define ray.
 * Represents a ray with an origin and direction in 3D space.
 * The direction will be normalized by default.
 */

class ray
{
private:
    point3 orig; ///< Origin point of the ray.
    vec3 dir;    ///< Direction vector of the ray.

public:
    /**
     * Default constructor.
     */
    ray() : orig(point3()), dir(vec3()) {}

    /**
     * Constructor that initializes the ray with a given origin and direction.
     * The direction is always normalized to maintain consistency.
     * Note that there are code in other places that rely on direction being a unit vector.
     *
     * @param origin The starting point of the ray.
     * @param direction The direction vector of the ray.
     * @param normalize If true, the direction vector will be normalized to a unit vector (default is true).
     */
    ray(const point3 &origin, const vec3 &direction) : orig(origin), dir(unit_vector(direction)) {}

    /**
     * Getter for the origin of the ray.
     *
     * @return The origin point of the ray.
     */
    const point3 &origin() const { return orig; }

    /**
     * Getter for the direction of the ray.
     *
     * @return The direction vector of the ray.
     */
    const vec3 &direction() const { return dir; }

    /**
     * Compute a point along the ray at a given distance from the origin.
     *
     * @param t Distance along the ray.
     * @return The point at distance t along the ray.
     */
    point3 at(double t) const
    {
        return orig + t * dir;
    }
};

/**
 * Output stream operator for ray.
 *
 * @param out Output stream.
 * @param r Ray to output.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &out, const ray &r)
{
    return out << "ray(origin: " << r.origin() << ", direction: " << r.direction() << ")";
}

/**
 * Equality operator for ray.
 *
 * @param r1 First ray.
 * @param r2 Second ray.
 * @return True if the origins and directions of the rays are equal, false otherwise.
 */
inline bool operator==(const ray &r1, const ray &r2)
{
    return r1.origin() == r2.origin() && r1.direction() == r2.direction();
}

/**
 * Inequality operator for ray.
 *
 * @param r1 First ray.
 * @param r2 Second ray.
 * @return True if the rays are not equal, false otherwise.
 */
inline bool operator!=(const ray &r1, const ray &r2)
{
    return !(r1 == r2);
}

#endif
