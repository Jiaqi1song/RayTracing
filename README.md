# CMU 15-618 Fall 2024 Course Project - Ray Tracing

#### Jiaqi Song (<jiaqison@andrew.cmu.edu>), Xinping Luo (<xinpingl@andrew.cmu.edu>)

We use the CPU start code from [Ray Tracing in One Weekend series](https://raytracing.github.io/). We plan to use OpenMP and CUDA to accelerate the rendering process.

## Check List

### CPU Renderer

- [x] Basic setup on CPU
- [x] Benchmark test case
- [x] Shared image data address
- [x] Use OpenMP and BVH to accelerate
- [x] Animation

### CUDA Renderer

- [x] Basic setup on CUDA 
- [x] Benchmark test case
- [ ] Monte Carlo sampling on CUDA 
- [ ] BVH acceleration on CUDA 
- [ ] Animation

## Compile and run the ray tracer (modify the parameters in the script)

```bash
bash render.sh
```

## Reference output image

### First Scene Animation (rotate + zoom)
![image](./images/animation1.gif)

### First Scene Animation (translate)
![image](./images/animation2.gif)

### First Scene Animation (bounce sphere)
![image](./images/animation3.gif)

### First Scene
![image](./images/first_scene.png)

### Cornell Box
![image](./images/cornell_box.png)

### Final scene
![image](./images/final_scene.png)

