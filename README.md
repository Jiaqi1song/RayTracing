# CMU 15-618 Fall 2024 Course Project - Ray Tracing

#### Jiaqi Song (<jiaqison@andrew.cmu.edu>), Xinping Luo (<xinpingl@andrew.cmu.edu>)

We use the CPU start code from [Ray Tracing in One Weekend series](https://raytracing.github.io/). We plan to use OpenMP and CUDA to accelerate the rendering process.

## TODO List

### CPU Renderer

- [x] Basic setup on CPU
- [x] Benchmark test case
- [x] Shared image data address
- [x] Use OpenMP and BVH to accelerate
- [x] Animation

### CUDA Renderer

- [ ] Basic setup on CUDA 
- [ ] Benchmark test case
- [ ] Monte Carlo sampling on CUDA 
- [ ] BVH acceleration on CUDA 
- [ ] Animation


## Reference output image

### Animation (rotate + zoom)
![image](./images/animation.gif)

### Cornell Box
![image](./images/cornell_box.png)

### Cornell Smoke
![image](./images/cornell_smoke.png)

### First Scene
![image](./images/first_scene.png)

### Final scene
![image](./images/final_scene.png)

