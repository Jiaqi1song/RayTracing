#ifndef CAMERA_H
#define CAMERA_H

#include <chrono>

#include "color.h"
#include "hittable.h"
#include "material.h"

/**
 * Class representing a camera in 3D space.
 * Responsible for generating rays through each pixel of the image and rendering the scene.
 */
class camera
{
private:
    // Default image size 1080 * 720.
    static constexpr int DEFAULT_IMAGE_WIDTH = 1080; ///< Default image width in pixels.
    static constexpr int DEFAULT_IMAGE_HEIGHT = 720; ///< Default image height in pixels.

    // Background colors for the gradient sky.
    color c1 = color(1.0, 1.0, 1.0); ///< Background color (white).
    color c2 = color(0.5, 0.7, 1.0); ///< Background color (light blue).

    double pixel_samples_scale; ///< Scale factor for averaging pixel colors.
    point3 camera_center;       ///< Location of the camera center in the scene.
    point3 pixel00_loc;         ///< Location of the center of the top-left pixel.
    vec3 pixel_delta_u;         ///< Offset to move from one pixel to the next horizontally.
    vec3 pixel_delta_v;         ///< Offset to move from one pixel to the next vertically.
    vec3 u, v, w;               ///< Camera coordinate frame basis vectors.
    vec3 defocus_disk_u;        ///< Defocus disk horizontal radius.
    vec3 defocus_disk_v;        ///< Defocus disk vertical radius.

    /**
     * Initializes the camera parameters based on given configuration.
     * Precomputes pixel positions, viewport vectors, and camera frame basis vectors.
     */
    void initialize()
    {
        pixel_samples_scale = 1.0 / samples_per_pixel; // Scale the sampled ray colors.

        // Set up viewport dimensions.
        camera_center = lookfrom;
        double theta = degrees_to_radians(vfov);
        double viewport_height =
            2.0 * std::tan(theta / 2) * focus_dist; // Adjust viewport height based on field of view.
        double viewport_width =
            viewport_height * (double(image_width) / image_height); // Adjust viewport width based on height.

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;
        vec3 viewport_v = viewport_height * -v;

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        point3 viewport_upper_left =
            camera_center - focus_dist * w - viewport_u / 2 - viewport_v / 2;
        // Move to the center of the upper left pixel.
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        double defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    /**
     * Computes the color seen along a ray in the scene.
     * Recursively traces rays to compute reflection, refraction, and shading effects.
     *
     * @param depth Maximum recursion depth.
     * @param r The ray to trace.
     * @param world The scene containing all hittable objects.
     * @return The color along the ray.
     */
    color ray_color(int depth, const ray &r, const hittable &world) const
    {
        // Meet max depth, end recursion.
        if (depth <= 0)
            return color();

        hit_record rec;
        if (world.hit(r, interval(0.001, infinity), rec))
        {
            ray scattered;
            color attenuation;
            if (rec.mat->scatter(r, rec, attenuation, scattered))
            {
                return attenuation * ray_color(depth - 1, scattered, world);
            }
            return color(); // No scattering, absorb the ray, end recursion.
        }

        // Hit background, end recursion.
        // Background gradient interpolation.
        double a = 0.5 * (r.direction().y() + 1.0);
        return lerp(c1, c2, a);
    }

    /**
     * Samples a point within the defocus disk for depth of field.
     *
     * @return A point within the defocus disk.
     */
    point3 defocus_disk_sample() const
    {
        vec3 vec = random_in_unit_disk();
        return camera_center + (vec[0] * defocus_disk_u) + (vec[1] * defocus_disk_v);
    }

    /**
     * Generates a ray passing through the given pixel center.
     *
     * @param pixel_center The center of the pixel through which the ray should pass.
     * @param r The generated ray to be populated.
     */
    void get_ray(point3 pixel_center, ray &r) const
    {
        point3 pixel_sample = pixel_center +
                              ((random_double() - 0.5) * pixel_delta_u) +
                              ((random_double() - 0.5) * pixel_delta_v);
        point3 ray_origin = defocus_angle <= 0 ? camera_center : defocus_disk_sample();
        r = ray(ray_origin, pixel_sample - ray_origin);
    }

    /**
     * Computes the color of a single pixel by averaging multiple samples.
     *
     * @param i Row index of the pixel.
     * @param j Column index of the pixel.
     * @param world The scene containing all hittable objects.
     * @return The averaged color for the pixel.
     */
    color compute_pixel_color(int i, int j, const hittable &world) const
    {
        color pixel_color;
        point3 current_pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        ray r;
        for (int sample = 0; sample < samples_per_pixel; sample++)
        {
            get_ray(current_pixel_center, r);
            pixel_color += ray_color(max_depth, r, world);
        }
        return pixel_samples_scale * pixel_color;
    }

public:
    int image_width = DEFAULT_IMAGE_WIDTH;   ///< Image width in pixels.
    int image_height = DEFAULT_IMAGE_HEIGHT; ///< Image height in pixels.
    int samples_per_pixel = 16;              ///< Number of samples per pixel.
    int max_depth = 10;                      ///< Maximum number of ray bounces.

    double vfov = 90;                  ///< Vertical field of view (in degrees).
    point3 lookfrom = point3(0, 0, 0); ///< Camera location.
    point3 lookat = point3(0, 0, -1);  ///< Point the camera is aimed at.
    vec3 vup = vec3(0, 1, 0);          ///< Up direction for the camera.

    double defocus_angle = 0; ///< Angle for defocus effect (depth of field).
    double focus_dist = 10;   ///< Distance to the plane of focus.

    /** Default constructor. */
    camera() {}

    /**
     * Constructor to initialize the camera with specific image dimensions.
     *
     * @param image_width Image width in pixels.
     * @param image_height Image height in pixels.
     */
    camera(int image_width, int image_height)
    {
        if (image_width > 100 && image_height > 100)
        {
            this->image_width = image_width;
            this->image_height = image_height;
        }
    }

    /**
     * Renders the scene by iterating over all pixels and calculating their colors.
     *
     * @param world The scene containing all hittable objects.
     */
    void render(const hittable &world)
    {
        initialize();

        // PPM header.
        std::cout << "P3\n"
                  << image_width << " " << image_height << "\n255\n";

        auto start_time = std::chrono::high_resolution_clock::now();
        int total_pixels = image_width * image_height;
        double total_row_time = 0;

        // Render image row by row.
        for (int j = 0; j < image_height; j++)
        {
            auto row_start_time = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < image_width; i++)
            {
                color pixel_color = compute_pixel_color(i, j, world);
                write_color(std::cout, pixel_color);
            }

            auto row_end_time = std::chrono::high_resolution_clock::now();
            auto elapsed_row_time = std::chrono::duration_cast<std::chrono::seconds>(row_end_time - row_start_time);
            total_row_time += elapsed_row_time.count();
            double avg_time_per_pixel = (total_row_time * 1000) / static_cast<double>((j + 1) * image_width);
            double avg_time_per_row = total_row_time / static_cast<double>(j + 1);

            std::clog << "\rLines remaining: " << (image_height - j - 1)
                      << " | Time for this row (s): " << std::fixed << std::setprecision(6) << double(elapsed_row_time.count())
                      << " | Avg time per pixel (ms): " << avg_time_per_pixel
                      << " | Avg time per row (s): " << avg_time_per_row << std::flush;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
        double avg_time_per_pixel = (total_row_time * 1000) / static_cast<double>(total_pixels);
        double avg_time_per_row = total_row_time / static_cast<double>(image_height);

        std::clog << "\rDone.                                                          "
                  << "                                                               \n";
        std::clog << "Total render time (min): " << double(total_time.count()) << "\n";
        std::clog << "Average time per row (s): " << avg_time_per_row << "\n";
        std::clog << "Average time per pixel (ms): " << avg_time_per_pixel << "\n";
    }
};

#endif
