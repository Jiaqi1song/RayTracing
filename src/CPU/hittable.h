#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "interval.h"

using std::make_shared;
using std::shared_ptr;
using std::vector;

/**
 * Define hit_record class to save important information
 * of a ray hitting a hittable or a list of hittables.
 *
 * Define hittable abstract class.
 * Override 'hit' function to judge if a ray hits the object of hittable.
 *
 * Define hittable_list class.
 * Often referred as world or part of world.
 * Deals with a list of hittable objects.
 */

// Defined in material.h.
class material;

/**
 * Stores information about a ray-object intersection.
 * Includes:
 * - t: The ray parameter at the intersection.
 * - hit_point: The intersection point.
 * - normal_vector: The surface normal at the intersection.
 * - front_face: Whether the ray hits the front face or back face.
 * - mat: Pointer to the material of the object hit.
 */
class hit_record
{
public:
    double t = 0.0;              ///< The parameter t at the intersection point along the ray.
    point3 hit_point = point3(); ///< The point of intersection.
    vec3 normal_vector = vec3(); ///< The surface normal at the intersection.
    bool front_face = false;     ///< Whether the intersection was with the front face.
    shared_ptr<material> mat;    ///< Material of the intersected object.

    /**
     * Determine whether the ray is hitting the front face or back face.
     * Adjust the normal vector to always point against the ray direction.
     * This is critical for lighting calculations.
     *
     * @param r The ray involved in the intersection.
     * @param outward_normal The normal vector pointing outward from the surface.
     */
    inline void set_face_normal(const ray &r, const vec3 &outward_normal)
    {
        // Sets the hit_record normal_vector and front_face flag.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.
        front_face = dot(r.direction(), outward_normal) < 0;
        // A front face occurs if the dot product is negative,
        // meaning the ray is coming from outside the surface.
        normal_vector = front_face ? outward_normal : -outward_normal;
    }

    /**
     * Log the details of the hit record for debugging purposes.
     */
    void log() const
    {
        std::clog << "t: " << t
                  << ", hit_point: " << hit_point
                  << ", normal_vector: " << normal_vector
                  << ", front_face: " << front_face
                  << "\n";
    }
};

/**
 * Abstract base class for all hittable objects.
 * Defines the interface for ray-object intersection tests.
 */
class hittable
{
public:
    /**
     * Virtual destructor for base class.
     *
     * Use a virtual destructor in a base class to ensure derived class destructors are called.
     * when you intend to delete derived class objects through a base class pointer.
     * Without a virtual destructor, only the base class destructor will run,
     * potentially causing resource leaks in derived classes.
     */
    virtual ~hittable() = default;

    /**
     * Pure virtual function to check if a ray hits the object.
     *
     * @param r The ray to test.
     * @param ray_t Interval of valid ray parameters, representing the allowed range for the `t` value.
     * @param rec The hit record to populate if a hit is detected.
     * @return True if the ray hits the object, otherwise false.
     */
    virtual bool hit(const ray &r, const interval &ray_t, hit_record &rec) const = 0;
};

/**
 * A list of hittable objects.
 * Implements the hittable interface to represent a collection of hittable objects.
 * What you actually use.
 */
class hittable_list : public hittable
{
public:
    vector<shared_ptr<hittable>> objects; ///< List of hittable objects.

    /**
     * Default constructor that initializes an empty hittable list.
     */
    hittable_list() : objects() {}

    /**
     * Constructor that initializes the list with a single hittable object.
     *
     * @param object A shared pointer to a hittable object to add to the list.
     */
    hittable_list(shared_ptr<hittable> object)
    {
        add(object);
    }

    /**
     * Constructor that initializes the list with a vector of hittable objects.
     *
     * @param objects A vector of shared pointers to hittable objects.
     */
    hittable_list(vector<shared_ptr<hittable>> objects) : objects(objects) {} // Copies the vector.

    /**
     * Clear the list of hittable objects.
     */
    void clear()
    {
        objects.clear();
    }

    /**
     * Add a hittable object to the list.
     *
     * @param object A shared pointer to a hittable object to add.
     */
    void add(shared_ptr<hittable> object)
    {
        if (objects.capacity() == objects.size())
        {
            objects.reserve(objects.size() * 1.5); // Reserve additional space
        }
        objects.push_back(object);
    }

    /**
     * Check if the ray hits any object in the list.
     *
     * @param r The ray to test.
     * @param ray_t Interval of valid ray parameters, representing the allowed range for the `t` value.
     * @param rec The hit record to populate if a hit is detected.
     * @return True if the ray hits any object in the list, otherwise false.
     */
    bool hit(const ray &r, const interval &ray_t, hit_record &rec) const override
    {
        hit_record tmp_rec;
        bool hit_anything = false;
        double closest_t = ray_t.max;

        for (const auto &object : objects)
        {
            if (object->hit(r, interval(ray_t.min, closest_t), tmp_rec))
            {
                hit_anything = true;
                closest_t = tmp_rec.t;
                rec = tmp_rec;
            }
        }

        return hit_anything;
    }
};

#endif
