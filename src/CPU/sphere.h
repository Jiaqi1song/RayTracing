#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"

/**
 * Define sphere, a hittable.
 */

/**
 * This class represents a 3D sphere that can be intersected by rays.
 * This is supposed to be a immutable class.
 * Use the provided 'hit' function to judge if a ray hits the shpere.
 */
class sphere : public hittable
{
private:
    point3 center;            ///< Center of the sphere.
    double radius;            ///< Radius of the sphere.
    shared_ptr<material> mat; ///< Material of the sphere.

public:
    /**
     * Constructor that initializes a sphere with the given center, radius, and material.
     *
     * @param center The center point of the sphere.
     * @param radius The radius of the sphere. Must be non-negative.
     * @param mat Shared pointer to the material of the sphere.
     * @throws std::invalid_argument if the radius is negative.
     */
    sphere(const point3 &center, double radius, shared_ptr<material> mat)
        : center(center), radius(std::fmax(0.0, radius)), mat(mat) {}

    /**
     * Get the center of the sphere.
     *
     * @return The center point of the sphere.
     */
    const point3 &get_center() const { return center; }

    /**
     * Get the radius of the sphere.
     *
     * @return The radius of the sphere.
     */
    const double &get_radius() const { return radius; }

    /**
     * Determine if a ray intersects with the sphere.
     *
     * @param r The ray to test for intersection.
     * @param ray_t The interval of valid ray parameters.
     * @param rec The hit record to populate if an intersection is detected.
     * @return True if the ray hits the sphere, otherwise false.
     */
    bool hit(const ray &r, const interval &ray_t, hit_record &rec) const override
    {
        // Solve the equation of whether the ray intersects with the sphere.
        vec3 ray_origin_to_sphere_center = center - r.origin();
        double a = r.direction().length_squared();
        double h = dot(r.direction(), ray_origin_to_sphere_center); // Simplified computation of b/2.
        double c = ray_origin_to_sphere_center.length_squared() - radius * radius;
        double discriminant = h * h - a * c;

        // If the discriminant is negative, there is no real root and hence no intersection.
        if (discriminant < 0.0)
        {
            return false;
        }

        // There is both tangential hit and closest hit, but both can be solved in one method.
        double sqrtd = std::sqrt(discriminant);
        // Find the nearest root that lies in the acceptable range.
        // In our navie case, the smaller t is, the point is closer.
        double root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root))
        {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
            {
                return false;
            }
        }

        // Populate the hit record with intersection details.
        rec.t = root;
        rec.hit_point = r.at(root);
        rec.mat = mat;
        vec3 outward_normal = (rec.hit_point - center) / radius; // This is a unit vector.
        rec.set_face_normal(r, outward_normal);
        // Set the surface normal such that it always points opposite to the ray's direction.
        // This is important for shading calculations to distinguish between front and back faces.
        return true;
    }
};

#endif
