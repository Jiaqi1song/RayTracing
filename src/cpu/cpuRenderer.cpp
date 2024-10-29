#include "utils.h"

#include "bvh.h"
#include "camera.h"
#include "constant_medium.h"
#include "hittable_list.h"
#include "material.h"
#include "quad.h"
#include "sphere.h"
#include "texture.h"

#include <iomanip> 
#include <iostream>
#include <chrono>

void first_scene(int image_width, double aspect_ratio, int samples_per_pixel, int max_depth, bool use_openmp, int num_threads) {
    hittable_list world;

    auto checker = make_shared<checker_texture>(0.32, color(.2, .3, .1), color(.9, .9, .9));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    world = hittable_list(make_shared<bvh_node>(world));

    // Light Sources
    auto empty_material = shared_ptr<material>();
    hittable_list lights;
    lights.add(make_shared<sphere>(point3(0,-1000,0), 1, empty_material));
    
    camera renderer;

    renderer.aspect_ratio      = aspect_ratio;
    renderer.image_width       = image_width;
    renderer.samples_per_pixel = samples_per_pixel;
    renderer.max_depth         = max_depth;
    renderer.background        = color(0.70, 0.80, 1.00);
    renderer.use_openmp        = use_openmp;
    renderer.num_threads       = num_threads;

    renderer.vfov     = 20;
    renderer.lookfrom = point3(13,2,3);
    renderer.lookat   = point3(0,0,0);
    renderer.vup      = vec3(0,1,0);

    renderer.defocus_angle = 0.6;
    renderer.focus_dist    = 10.0;

    renderer.render(world, lights);
}


void cornell_box(int image_width, double aspect_ratio, int samples_per_pixel, int max_depth, bool use_openmp, int num_threads) {
    hittable_list world;

    auto red   = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(15, 15, 15));

    // Cornell box sides
    world.add(make_shared<quad>(point3(555,0,0), vec3(0,0,555), vec3(0,555,0), green));
    world.add(make_shared<quad>(point3(0,0,555), vec3(0,0,-555), vec3(0,555,0), red));
    world.add(make_shared<quad>(point3(0,555,0), vec3(555,0,0), vec3(0,0,555), white));
    world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,0,-555), white));
    world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));

    // Light
    world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

    // Box
    shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265,0,295));
    world.add(box1);

    // Glass Sphere
    auto glass = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(190,90,190), 90, glass));

    // Light Sources
    auto empty_material = shared_ptr<material>();
    hittable_list lights;
    lights.add(
        make_shared<quad>(point3(343,554,332), vec3(-130,0,0), vec3(0,0,-105), empty_material));
    lights.add(make_shared<sphere>(point3(190, 90, 190), 90, empty_material));

    camera renderer;

    renderer.aspect_ratio      = aspect_ratio;
    renderer.image_width       = image_width;
    renderer.samples_per_pixel = samples_per_pixel;
    renderer.max_depth         = max_depth;
    renderer.background        = color(0,0,0);
    renderer.use_openmp        = use_openmp;
    renderer.num_threads       = num_threads;

    renderer.vfov     = 40;
    renderer.lookfrom = point3(278, 278, -800);
    renderer.lookat   = point3(278, 278, 0);
    renderer.vup      = vec3(0, 1, 0);

    renderer.defocus_angle = 0;

    renderer.render(world, lights);
}

void cornell_smoke(int image_width, double aspect_ratio, int samples_per_pixel, int max_depth, bool use_openmp, int num_threads) {
    hittable_list world;

    auto red   = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(15, 15, 15));

    // Cornell box sides
    world.add(make_shared<quad>(point3(555,0,0), vec3(0,0,555), vec3(0,555,0), green));
    world.add(make_shared<quad>(point3(0,0,555), vec3(0,0,-555), vec3(0,555,0), red));
    world.add(make_shared<quad>(point3(0,555,0), vec3(555,0,0), vec3(0,0,555), white));
    world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,0,-555), white));
    world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));

    // Light
    world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

    shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265,0,295));

    shared_ptr<hittable> box2 = box(point3(0,0,0), point3(165,165,165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, vec3(130,0,65));

    world.add(make_shared<constant_medium>(box1, 0.01, color(0,0,0)));
    world.add(make_shared<constant_medium>(box2, 0.01, color(1,1,1)));

    // Light Sources
    auto empty_material = shared_ptr<material>();
    hittable_list lights;
    lights.add(make_shared<quad>(point3(343,554,332), vec3(-130,0,0), vec3(0,0,-105), empty_material));

    camera renderer;

    renderer.aspect_ratio      = aspect_ratio;
    renderer.image_width       = image_width;
    renderer.samples_per_pixel = samples_per_pixel;
    renderer.max_depth         = max_depth;
    renderer.background        = color(0,0,0);
    renderer.use_openmp        = use_openmp;
    renderer.num_threads       = num_threads;

    renderer.vfov     = 40;
    renderer.lookfrom = point3(278, 278, -800);
    renderer.lookat   = point3(278, 278, 0);
    renderer.vup      = vec3(0, 1, 0);

    renderer.defocus_angle = 0;

    renderer.render(world, lights);
}

void final_scene(int image_width, double aspect_ratio, int samples_per_pixel, int max_depth, bool use_openmp, int num_threads) {
    hittable_list boxes1;
    auto ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));

    int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_double(1,101);
            auto z1 = z0 + w;

            boxes1.add(box(point3(x0,y0,z0), point3(x1,y1,z1), ground));
        }
    }

    hittable_list world;

    world.add(make_shared<bvh_node>(boxes1));

    auto light = make_shared<diffuse_light>(color(15, 15, 15));
    world.add(make_shared<quad>(point3(123,554,147), vec3(300,0,0), vec3(0,0,265), light));

    auto center1 = point3(400, 400, 200);
    auto center2 = center1 + vec3(30,0,0);
    auto sphere_material = make_shared<lambertian>(color(0.7, 0.3, 0.1));
    world.add(make_shared<sphere>(center1, center2, 50, sphere_material));

    world.add(make_shared<sphere>(point3(260, 150, 45), 50, make_shared<dielectric>(1.5)));
    world.add(make_shared<sphere>(
        point3(0, 150, 145), 50, make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)
    ));

    auto boundary = make_shared<sphere>(point3(360,150,145), 70, make_shared<dielectric>(1.5));
    world.add(boundary);
    world.add(make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));
    boundary = make_shared<sphere>(point3(0,0,0), 5000, make_shared<dielectric>(1.5));
    world.add(make_shared<constant_medium>(boundary, .0001, color(1,1,1)));

    auto emat = make_shared<lambertian>(make_shared<image_texture>("earthmap.jpg"));
    world.add(make_shared<sphere>(point3(400,200,400), 100, emat));
    auto pertext = make_shared<noise_texture>(0.2);
    world.add(make_shared<sphere>(point3(220,280,300), 80, make_shared<lambertian>(pertext)));

    hittable_list boxes2;
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2.add(make_shared<sphere>(point3::random(0,165), 10, white));
    }

    world.add(make_shared<translate>(
        make_shared<rotate_y>(
            make_shared<bvh_node>(boxes2), 15),
            vec3(-100,270,395)
        )
    );

    // Light Sources
    auto empty_material = shared_ptr<material>();
    hittable_list lights;
    lights.add(make_shared<quad>(point3(253,554,253), vec3(-300,0,0), vec3(0,0,-265), empty_material));
    
    camera renderer;

    renderer.aspect_ratio      = aspect_ratio;
    renderer.image_width       = image_width;
    renderer.samples_per_pixel = samples_per_pixel;
    renderer.max_depth         = max_depth;
    renderer.background        = color(0,0,0);
    renderer.use_openmp        = use_openmp;
    renderer.num_threads       = num_threads;

    renderer.vfov     = 40;
    renderer.lookfrom = point3(478, 278, -600);
    renderer.lookat   = point3(278, 278, 0);
    renderer.vup      = vec3(0,1,0);

    renderer.defocus_angle = 0;

    renderer.render(world, lights);
}


int main() {
    // Scene selection
    int scene = 2;         

    // Acceleration technique selection
    bool use_openmp = false;
    int num_threads = 8;

    // Hyperparameters
    int image_width = 600;                // Rendered image width in pixel count
    double aspect_ratio = 1.0;            // Ratio of image width over height
    int samples_per_pixel = 30;           // Count of random samples for each pixel
    int max_depth = 20;                   // Maximum number of ray bounces into scene

    auto startTime = std::chrono::high_resolution_clock::now();
    switch (scene) {
        case 1:  first_scene(image_width, aspect_ratio, samples_per_pixel, max_depth, use_openmp, num_threads);     break;
        case 2:  cornell_box(image_width, aspect_ratio, samples_per_pixel, max_depth, use_openmp, num_threads);     break;
        case 3:  cornell_smoke(image_width, aspect_ratio, samples_per_pixel, max_depth, use_openmp, num_threads);   break;
        default: final_scene(image_width, aspect_ratio, samples_per_pixel, max_depth, use_openmp, num_threads);     break;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;

    std::clog << "Overall Rendering Time: " << std::fixed << std::setprecision(4) << duration.count() << " seconds\n"; 
}
