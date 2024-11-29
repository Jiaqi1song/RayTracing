#include "camera.h"
#include "sphere.h"
#include "texture.h"
#include "quad.h"
#include "bvh.h"

#include <chrono>

#define MAX_OBJS 500

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '"
                  << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void create_world1(hittable **d_list, hittable_list **d_world, camera **cam, int image_width,
                             int image_height, curandState *devStates, int samples_per_pixel, int max_depth, bool use_bvh)
{
    curandState *local_rand_state = &devStates[0];

    auto checker = new checker_texture(0.32, color(.8, .1, .1), color(.9, .9, .9));

    d_list[0] = new sphere(point3(0.0f, -1000.0f, -1.0f), 1000.0f, new lambertian(checker));
    int i = 1;
    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            float choose_mat = random_float(local_rand_state);
            point3 center(a + random_float(local_rand_state), 0.2f, b + random_float(local_rand_state));
            if (choose_mat < 0.8f)
            {
                d_list[i++] =
                    new sphere(center, 0.2f,
                               new lambertian(color(random_float(local_rand_state) * random_float(local_rand_state),
                                                    random_float(local_rand_state) * random_float(local_rand_state),
                                                    random_float(local_rand_state) * random_float(local_rand_state))));
            }
            else if (choose_mat < 0.95f)
            {
                d_list[i++] = new sphere(center, 0.2f,
                                         new metal(color(0.5f * (1.0f + random_float(local_rand_state)),
                                                         0.5f * (1.0f + random_float(local_rand_state)),
                                                         0.5f * (1.0f + random_float(local_rand_state))),
                                                   0.5f * random_float(local_rand_state)));
            }
            else
            {
                d_list[i++] = new sphere(center, 0.2f, new dielectric(1.5));
            }
        }
    }
    d_list[i++] = new sphere(point3(0.0f, 1.0f, 0.0f), 1.0f, new dielectric(1.5f));
    d_list[i++] = new sphere(point3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(color(0.4f, 0.2f, 0.1f)));
    d_list[i++] = new sphere(point3(4.0f, 1.0f, 0.0f), 1.0f, new metal(color(0.7f, 0.6f, 0.5f), 0.0f));
    
    *d_world = new hittable_list(d_list, i);

    *cam = new camera(image_width, image_height, samples_per_pixel, max_depth, 20.0f, point3(13.0f, 2.0f, 3.0f), point3(0.0f, 0.0f, 0.0f),
                      vec3(0.0f, 1.0f, 0.0f), 0.6f, 10.0f, color(0.70, 0.80, 1.00));
}

__global__ void create_world2(hittable **d_list, hittable_list **d_world, camera **cam, int image_width,
                             int image_height, curandState *devStates, int samples_per_pixel, int max_depth, bool use_bvh)
{
    // Cornell box sides
    int i = 0;
    d_list[i++] = new quad(point3(555,0,0), vec3(0,0,555), vec3(0,555,0), new lambertian(color(.12, .45, .15)));
    d_list[i++] = new quad(point3(0,0,555), vec3(0,0,-555), vec3(0,555,0), new lambertian(color(.65, .05, .05)));
    d_list[i++] = new quad(point3(0,555,0), vec3(555,0,0), vec3(0,0,555), new lambertian(color(.73, .73, .73)));
    d_list[i++] = new quad(point3(0,0,555), vec3(555,0,0), vec3(0,0,-555), new lambertian(color(.73, .73, .73)));
    d_list[i++] = new quad(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), new lambertian(color(.73, .73, .73)));

    // Light
    d_list[i++] = new quad(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), new diffuse_light(color(15, 15, 15)));

    // Box
    d_list[i++] = new quad(point3(265, 0, 295), vec3(159.38, 0, -42.71), vec3(42.71, 0, 159.38), new lambertian(color(.73, .73, .73)));
    d_list[i++] = new quad(point3(424.38, 0, 252.29), vec3(42.71, 0, 159.38), vec3(0, 330, 0), new lambertian(color(.73, .73, .73)));
    d_list[i++] = new quad(point3(467.08, 0, 411.67), vec3(-159.38, 0, 42.71), vec3(0, 330, 0), new lambertian(color(.73, .73, .73))); 
    d_list[i++] = new quad(point3(307.71, 0, 454.38), vec3(-42.71, 0, -159.38), vec3(0, 330, 0), new lambertian(color(.73, .73, .73)));
    d_list[i++] = new quad(point3(265, 330, 295), vec3(159.38, 0, -42.71), vec3(42.71, 0, 159.38), new lambertian(color(.73, .73, .73))); 
    d_list[i++] = new quad(point3(265, 0, 295), vec3(159.38, 0, -42.71), vec3(0, 330, 0), new lambertian(color(.73, .73, .73)));   


    // Glass Sphere
    d_list[i++] = new sphere(point3(190.0f,90.0f,190.0f), 90.0f, new dielectric(1.5f));
    
    *d_world = new hittable_list(d_list, i);

    *cam = new camera(image_width, image_height, samples_per_pixel, max_depth, 40.0f, point3(278.0f, 278.0f, -800.0f), point3(278.0f, 278.0f, 0.0f),
                      vec3(0.0f, 1.0f, 0.0f), 0.0f, 10.0f, color(0,0,0));
}

__global__ void free_world(hittable **d_list, hittable_list **d_world, camera **d_camera)
{
    for (int i = 0; i < (*d_world)->obj_num; i++)
    {
        if (d_list[i]->get_type() == HittableType::SPHERE) {
            delete ((sphere*)d_list[i])->get_mat();     
        } else if (d_list[i]->get_type() == HittableType::QUAD) {
            delete ((quad*)d_list[i])->get_mat();          
        }
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__global__ void call_render(hittable_list **d_world, camera **cam, int image_width, int image_height, uint8_t *output,
                            curandState *devStates)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= image_width) || (j >= image_height))
        return;

    curandState *local_rand_state = &devStates[j * image_width + i];

    (*cam)->render(d_world, i, j, local_rand_state, output);
}

int main()
{
    int image_width = 1080;
    int image_height = 720;
    int samples_per_pixel = 200;
    int max_depth = 20;
    int scene = 1;
    bool use_bvh = false;

    int total_pixels = image_width * image_height;

    curandState *devStates;
    checkCudaErrors(cudaMalloc((void **)&devStates, total_pixels * sizeof(curandState)));
    hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, MAX_OBJS * sizeof(hittable *)));
    hittable_list **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable_list *)));
    uint8_t *d_output;
    checkCudaErrors(cudaMalloc((void **)&d_output, total_pixels * 3 * sizeof(uint8_t)));
    camera **cam;
    checkCudaErrors(cudaMalloc((void **)&cam, sizeof(camera)));

    int blockdimx = 16;
    int blockdimy = 16;
    dim3 gridSize((image_width + blockdimx - 1) / blockdimx, (image_height + blockdimy - 1) / blockdimy);
    dim3 blockSize(blockdimx, blockdimy);

    auto start_time = std::chrono::high_resolution_clock::now();

    unsigned long seed = static_cast<unsigned long>(start_time.time_since_epoch().count());
    init_random_state<<<gridSize, blockSize>>>(devStates, image_width, image_height, seed);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    switch (scene) {
        case 1: create_world1<<<1, 1>>>(d_list, d_world, cam, image_width, image_height, devStates, samples_per_pixel, max_depth, use_bvh); break;
        case 2: create_world2<<<1, 1>>>(d_list, d_world, cam, image_width, image_height, devStates, samples_per_pixel, max_depth, use_bvh); break;
    }
    
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    call_render<<<gridSize, blockSize>>>(d_world, cam, image_width, image_height, d_output, devStates);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto render_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    float total_time = render_time.count();
    float avg_time_per_pixel = total_time / static_cast<float>(total_pixels);
    float avg_time_per_row = total_time / static_cast<float>(image_height);

    std::clog << "Total render time (ms): " << total_time << "\n";
    std::clog << "Average time per row (ms): " << avg_time_per_row << "\n";
    std::clog << "Average time per pixel (ms): " << avg_time_per_pixel << "\n";

    uint8_t *h_output = new uint8_t[image_width * image_height * 3];
    checkCudaErrors(
        cudaMemcpy(h_output, d_output, image_width * image_height * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int i = 0; i < image_height; i++)
    { // Row
        std::clog << "\rScanlines remaining: " << (image_height - 1) << " " << std::flush;
        for (int j = 0; j < image_width; j++)
        { // Column
            int start_write_index = 3 * (i * image_width + j);
            write_color(std::cout, h_output[start_write_index], h_output[start_write_index + 1],
                        h_output[start_write_index + 2]);
        }
    }

    std::clog << "\rDone.                   \n";

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(devStates));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(cam));
    cudaDeviceReset();
    delete[] h_output;
}
