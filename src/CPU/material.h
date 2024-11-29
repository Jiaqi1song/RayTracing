#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "hittable.h"

/**
 * Define the base class of materials.
 * Define materials that can given to hittable objects.
 */

/**
 * Base class of materials.
 */
class material
{
public:
    /** Virtual destructor to allow derived classes to clean up properly. */
    virtual ~material() = default;

    /**
     * Pure virtual function to determine if a ray scatters when hitting a material.
     * Must be overridden by derived classes.
     *
     * @param r_in The incoming ray.
     * @param rec The hit record that contains information about the intersection.
     * @param attenuation The color attenuation factor resulting from scattering.
     * @param r_out The outgoing scattered ray.
     * @return True if the ray is scattered, otherwise false.
     */
    virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &r_out) const = 0;
};

/**
 * Lambertian material that represents diffuse reflection.
 * The scatter direction is randomized, resulting in a matte appearance.
 */
class lambertian : public material
{
private:
    color albedo; ///< The albedo represents the diffuse reflectivity of the surface.

public:
    /**
     * Constructor to initialize the Lambertian material with a specific albedo.
     *
     * @param albedo The color that represents the reflectivity of the surface.
     */
    lambertian(const color &albedo) : albedo(albedo) {}

    /**
     * Scatter function to determine how the ray interacts with the Lambertian material.
     *
     * @param r_in The incoming ray.
     * @param rec The hit record containing intersection information.
     * @param attenuation The resulting attenuation of the color.
     * @param r_out The resulting scattered ray.
     * @return True, indicating that the ray always scatters.
     */
    bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &r_out) const override
    {
        vec3 scatter_direction = rec.normal_vector + random_unit_vector();
        while (scatter_direction.near_zero())
            scatter_direction = rec.normal_vector + random_unit_vector();
        r_out = ray(rec.hit_point, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

/**
 * Metal material that represents reflective surfaces.
 * Can include fuzziness to make the reflection imperfect.
 */
class metal : public material
{
private:
    color albedo; ///< The color representing the reflectivity of the metal.
    double fuzz;  ///< Fuzziness factor to make the reflection imperfect, range [0, 1].

public:
    /**
     * Constructor to initialize the metal material with an albedo and fuzziness.
     *
     * @param albedo The color representing the metal's reflectivity.
     * @param fuzz The fuzziness of the reflection, clamped to [0, 1].
     */
    metal(const color &albedo, double fuzz) : albedo(albedo), fuzz((std::clamp(fuzz, 0.0, 1.0))) {}

    /**
     * Scatter function to determine how the ray interacts with the metal material.
     *
     * @param r_in The incoming ray.
     * @param rec The hit record containing intersection information.
     * @param attenuation The resulting attenuation of the color.
     * @param r_out The resulting reflected ray.
     * @return True if the ray is reflected, false otherwise.
     */
    bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &r_out) const override
    {
        vec3 reflect_direction = reflect(r_in.direction(), rec.normal_vector);
        reflect_direction = unit_vector(reflect_direction) + fuzz * random_unit_vector(); // Add fuzz for imperfect reflection
        r_out = ray(rec.hit_point, reflect_direction);
        attenuation = albedo;
        return (dot(reflect_direction, rec.normal_vector) > 0); // Only reflect if the direction is above the surface.
    }
};

/**
 * Dielectric material that represents transparent surfaces with refraction.
 * Uses Snell's law to calculate refraction and Schlick's approximation for reflectance.
 */
class dielectric : public material
{
private:
    double refraction_index;                              ///< Index of refraction for the material.
    color dielectrics_attenuation = color(1.0, 1.0, 1.0); ///< The attenuation for dielectric materials (white).

    /**
     * Calculate reflectance using Schlick's approximation.
     *
     * @param cosine The cosine of the angle between the ray and the surface normal.
     * @param refraction_index The index of refraction of the material.
     * @return The probability of reflection.
     */
    static double reflectance(double cosine, double refraction_index)
    {
        // Use Schlick's approximation for reflectance.
        double r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * std::pow((1 - cosine), 5);
    }

public:
    /**
     * Constructor to initialize the dielectric material with a specific refraction index.
     *
     * @param refraction_index The index of refraction of the dielectric material.
     */
    dielectric(double refraction_index) : refraction_index(refraction_index) {}

    /**
     * Scatter function to determine how the ray interacts with the dielectric material.
     * Can either reflect or refract depending on Snell's law and reflectance.
     *
     * @param r_in The incoming ray.
     * @param rec The hit record containing intersection information.
     * @param attenuation The resulting attenuation of the color (usually white for transparency).
     * @param r_out The resulting scattered ray (reflected or refracted).
     * @return True, indicating that the ray either reflects or refracts.
     */
    bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &r_out) const override
    {
        attenuation = dielectrics_attenuation;
        double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;

        double cos_theta = std::fmin(dot(-r_in.direction(), rec.normal_vector), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_double())
            direction = reflect(r_in.direction(), rec.normal_vector);
        else
            direction = refract(r_in.direction(), rec.normal_vector, ri);

        r_out = ray(rec.hit_point, direction);
        return true;
    }
};

#endif
