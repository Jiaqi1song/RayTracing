#ifndef CAMERA_H
#define CAMERA_H

#include "utils.h"
#include "hittable.h"
#include "pdf.h"
#include "material.h"
#include <omp.h>
#include <string>
#include <vector>

enum CameraMovement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

class camera {
  public:
    double aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene
    color  background;               // Scene background color
    bool   use_openmp = false;       // Use OpenMP
    int    num_threads = 8;          
    Image* image = NULL;
    char filepath[1024];
    std::string filename = "image.ppm";
    
    double vfov     = 90;              // Vertical view angle (field of view)
    point3 lookfrom = point3(0,0,0);   // Point camera is looking from
    point3 lookat   = point3(0,0,-1);  // Point camera is looking at
    vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

    int animation_sample = int(2 * pi / delta_phi);
    int animation_method = 0;
    bool use_bvh = true;

    void render_animation(const hittable& world, const hittable& lights) {
        
        if (animation_method = 0) {
            vec3 direction = lookfrom - lookat;
            double theta = std::acos(direction.y() / direction.length());
            double phi = std::atan2(direction.x(), direction.z());
            double zoom_scale = 1.005;

            std::clog << "\rTotal frames: " << animation_sample << ": \n";
            for (int frame = 0; frame < animation_sample; frame++) {
                
                rotate(theta, phi);
                zoom(zoom_scale);
                initialize();

                // Output file path
                sprintf(filepath, "./images/animation/image%d.ppm", frame);

                std::clog << "\rStart Rendering Frame " << frame << ": \n";
                if (use_openmp) {
                    omp_set_num_threads(num_threads);
                    std::cout << "Max number of threads: " << omp_get_max_threads() << std::endl;
                    std::clog << "\rStart Rendering " << image_height * image_width << " pixels on CPU with OpenMP...\n";
                } else {
                    std::clog << "\rStart Rendering " << image_height * image_width << " pixels on CPU without OpenMP...\n";
                }
                
                #pragma omp parallel for if(use_openmp) collapse(2) schedule(dynamic)
                for (int j = 0; j < image_height; j++) {
                    for (int i = 0; i < image_width; i++) {
                        if (!use_openmp) std::clog << "\rPixels remaining: " << (image_width * image_height - (j * image_width + i)) << ' ' << std::flush;
                        color pixel_color(0,0,0);
                        for (int s_j = 0; s_j < sqrt_spp; s_j++) {
                            for (int s_i = 0; s_i < sqrt_spp; s_i++) {
                                ray r = get_ray(i, j, s_i, s_j);
                                pixel_color += ray_color(r, max_depth, world, lights);
                            }
                        }

                        int* imgPtr = &image->data[3 * (j * image_width + i)];
                        write_color(imgPtr, pixel_samples_scale * pixel_color);
                    }
                }
                
                std::clog << "\rRendering Frame " << frame << " Done.       \n";
                writePPMImage(image, filepath);

                phi += delta_phi; 
                if (phi >= 2 * pi) phi -= 2 * pi;
                theta -= 0.01;
                if (theta <= 0) theta = 0;
            }
        } else if (animation_method = 1) {
            double step_scale = 0.6;
            animation_sample = 42;

            std::clog << "\rTotal frames: " << animation_sample << ": \n";
            for (int frame = 0; frame < animation_sample; frame++) {
                
                if (frame < 7) {
                    translate(FORWARD, step_scale);
                } else if (frame < 14) {
                    translate(BACKWARD, step_scale);
                } else if (frame < 21) {
                    translate(LEFT, step_scale);
                } else if (frame < 28) {
                    translate(RIGHT, step_scale);
                } else if (frame < 35) {
                    translate(UP, step_scale);
                } else if (frame < 42) {
                    translate(DOWN, step_scale);
                }

                initialize();

                // Output file path
                sprintf(filepath, "./images/animation/image%d.ppm", frame);

                std::clog << "\rStart Rendering Frame " << frame << ": \n";
                if (use_openmp) {
                    omp_set_num_threads(num_threads);
                    std::cout << "Max number of threads: " << omp_get_max_threads() << std::endl;
                    std::clog << "\rStart Rendering " << image_height * image_width << " pixels on CPU with OpenMP...\n";
                } else {
                    std::clog << "\rStart Rendering " << image_height * image_width << " pixels on CPU without OpenMP...\n";
                }
                
                #pragma omp parallel for if(use_openmp) collapse(2) schedule(dynamic)
                for (int j = 0; j < image_height; j++) {
                    for (int i = 0; i < image_width; i++) {
                        if (!use_openmp) std::clog << "\rPixels remaining: " << (image_width * image_height - (j * image_width + i)) << ' ' << std::flush;
                        color pixel_color(0,0,0);
                        for (int s_j = 0; s_j < sqrt_spp; s_j++) {
                            for (int s_i = 0; s_i < sqrt_spp; s_i++) {
                                ray r = get_ray(i, j, s_i, s_j);
                                pixel_color += ray_color(r, max_depth, world, lights);
                            }
                        }

                        int* imgPtr = &image->data[3 * (j * image_width + i)];
                        write_color(imgPtr, pixel_samples_scale * pixel_color);
                    }
                }
                
                std::clog << "\rRendering Frame " << frame << " Done.       \n";
                writePPMImage(image, filepath);
            }
        } 
    }


    void render(const hittable& world, const hittable& lights) {
        initialize();

        // Output file path
        sprintf(filepath, "./images/%s", filename.c_str());

        if (use_openmp) {
            omp_set_num_threads(num_threads);
            std::cout << "Max number of threads: " << omp_get_max_threads() << std::endl;
            std::clog << "\rStart Rendering " << image_height * image_width << " pixels on CPU with OpenMP...\n";
        } else {
            std::clog << "\rStart Rendering " << image_height * image_width << " pixels on CPU without OpenMP...\n";
        }
        
        #pragma omp parallel for if(use_openmp) collapse(2) schedule(dynamic)
        for (int j = 0; j < image_height; j++) {
            for (int i = 0; i < image_width; i++) {
                if (!use_openmp) std::clog << "\rPixels remaining: " << (image_width * image_height - (j * image_width + i)) << ' ' << std::flush;
                color pixel_color(0,0,0);
                for (int s_j = 0; s_j < sqrt_spp; s_j++) {
                    for (int s_i = 0; s_i < sqrt_spp; s_i++) {
                        ray r = get_ray(i, j, s_i, s_j);
                        pixel_color += ray_color(r, max_depth, world, lights);
                    }
                }

                int* imgPtr = &image->data[3 * (j * image_width + i)];
                write_color(imgPtr, pixel_samples_scale * pixel_color);
            }
        }
        
        std::clog << "\rRendering Done. Writing image to " << filename.c_str() << ".         \n";
        writePPMImage(image, filepath);
    }

  private:
    int    image_height;         // Rendered image height
    double pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    int    sqrt_spp;             // Square root of number of samples per pixel
    double recip_sqrt_spp;       // 1 / sqrt_spp
    point3 center;               // Camera center
    point3 pixel00_loc;          // Location of pixel 0, 0
    vec3   pixel_delta_u;        // Offset to pixel to the right
    vec3   pixel_delta_v;        // Offset to pixel below
    vec3   u, v, w;              // Camera frame basis vectors
    vec3   defocus_disk_u;       // Defocus disk horizontal radius
    vec3   defocus_disk_v;       // Defocus disk vertical radius

    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        // Allocate image
        if (image) delete image;
        image = new Image(image_width, image_height);
        image->clear(0,0,0);

        sqrt_spp = int(std::sqrt(samples_per_pixel));
        pixel_samples_scale = 1.0 / (sqrt_spp * sqrt_spp);
        recip_sqrt_spp = 1.0 / sqrt_spp;

        center = lookfrom;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta/2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    ray get_ray(int i, int j, int s_i, int s_j) const {
        // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j for stratified sample square s_i, s_j.

        auto offset = sample_square_stratified(s_i, s_j);
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        auto ray_direction = pixel_sample - ray_origin;
        auto ray_time = random_double();

        return ray(ray_origin, ray_direction, ray_time);
    }

    vec3 sample_square_stratified(int s_i, int s_j) const {
        // Returns the vector to a random point in the square sub-pixel specified by grid
        // indices s_i and s_j, for an idealized unit square pixel [-.5,-.5] to [+.5,+.5].

        auto px = ((s_i + random_double()) * recip_sqrt_spp) - 0.5;
        auto py = ((s_j + random_double()) * recip_sqrt_spp) - 0.5;

        return vec3(px, py, 0);
    }

    vec3 sample_square() const {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return vec3(random_double() - 0.5, random_double() - 0.5, 0);
    }

    vec3 sample_disk(double radius) const {
        // Returns a random point in the unit (radius 0.5) disk centered at the origin.
        return radius * random_in_unit_disk();
    }

    point3 defocus_disk_sample() const {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    color ray_color(const ray& r, int depth, const hittable& world, const hittable& lights)
    const {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0,0,0);

        hit_record rec;

        // If the ray hits nothing, return the background color.
        if (!world.hit(r, interval(0.001, infinity), rec))
            return background;

        scatter_record srec;
        color color_from_emission = rec.mat->emitted(r, rec, rec.u, rec.v, rec.p);

        if (!rec.mat->scatter(r, rec, srec))
            return color_from_emission;

        if (srec.skip_pdf) {
            return srec.attenuation * ray_color(srec.skip_pdf_ray, depth-1, world, lights);
        }

        auto light_ptr = make_shared<hittable_pdf>(lights, rec.p);
        mixture_pdf p(light_ptr, srec.pdf_ptr);

        ray scattered = ray(rec.p, p.generate(), r.time());
        auto pdf_value = p.value(scattered.direction());

        double scattering_pdf = rec.mat->scattering_pdf(r, rec, scattered);

        color sample_color = ray_color(scattered, depth-1, world, lights);
        color color_from_scatter =
            (srec.attenuation * scattering_pdf * sample_color) / pdf_value;

        return color_from_emission + color_from_scatter;
    }

    void rotate(double theta, double phi) {
        double radialDistance = (lookfrom - lookat).length();
        lookfrom = vec3(
            radialDistance * std::sin(theta) * std::sin(phi),
            radialDistance * std::cos(theta),
            radialDistance * std::sin(theta) * std::cos(phi)
        ) + lookat;
    }

    void zoom(double zoom_scale) {
        vec3 direction = lookfrom - lookat;
        lookfrom = direction * zoom_scale + lookat;
        focus_dist *= zoom_scale;
    }

    void translate(CameraMovement direction, double step_scale) {
        if (direction == FORWARD) {
            lookfrom += w * step_scale;
            lookat += w * step_scale;
            focus_dist += step_scale;
        }
        if (direction == BACKWARD) {
            lookfrom += w * -step_scale;
            lookat += w * -step_scale;
            focus_dist -= step_scale;
        }
        if (direction == LEFT) {
            lookfrom += u * -step_scale;
            lookat += u * -step_scale;
        }
        if (direction == RIGHT) {
            lookfrom += u * step_scale;
            lookat += u * step_scale;
        }
        if (direction == UP) {
            lookfrom += v * step_scale;
            lookat += v * step_scale;
        }
        if (direction == DOWN) {
            lookfrom += v * -step_scale;
            lookat += v * -step_scale;
        }
    }

};


#endif
