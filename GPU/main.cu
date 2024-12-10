#include "camera.h"
#include "sphere.h"
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

__global__ void create_world(hittable **d_list, hittable_list **d_world, BVH_Node *bvh_data, camera **cam, int image_width,
                             int image_height, curandState *devStates, bool use_bvh)
{
    curandState *local_rand_state = &devStates[0];

    d_list[0] = new sphere(point3(0.0f, -1000.0f, -1.0f), 1000.0f, new lambertian(color(0.5f, 0.5f, 0.5f)));
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

    if (use_bvh) {
        d_list[0] = build_bvh_node(d_list, bvh_data, i, local_rand_state);
        *d_world = new hittable_list(d_list, 1);
    } else {
        *d_world = new hittable_list(d_list, i);
    }

    *cam = new camera(image_width, image_height, 100, 25, 20.0f, point3(13.0f, 2.0f, 3.0f), point3(0.0f, 0.0f, 0.0f),
                      vec3(0.0f, 1.0f, 0.0f), 0.6f, 10.0f);
}

__global__ void free_world(hittable **d_list, hittable_list **d_world, camera **d_camera)
{
    for (int i = 0; i < (*d_world)->obj_num; i++)
    {
        delete ((sphere *)d_list[i])->get_mat();
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

    int total_pixels = image_width * image_height;

    bool use_bvh = true;

    curandState *devStates;
    checkCudaErrors(cudaMalloc((void **)&devStates, total_pixels * sizeof(curandState)));
    BVH_Node *bvh_data;
    checkCudaErrors(cudaMalloc((void **)&bvh_data, MAX_OBJS * sizeof(BVH_Node)));
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

    create_world<<<1, 1>>>(d_list, d_world, bvh_data, cam, image_width, image_height, devStates, use_bvh);
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
    checkCudaErrors(cudaFree(bvh_data));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(devStates));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(cam));
    cudaDeviceReset();
    delete[] h_output;
}
