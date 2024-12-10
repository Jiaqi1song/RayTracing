#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.h"
#include "interval.h"
#include "ray.h"

// Static stack for hit traversal
template <typename T, size_t N> struct StaticStack
{
    __device__ StaticStack() : size_(0) {}

    __device__ void push(const T &value) { data_[size_++] = value; }

    __device__ T pop() { return data_[--size_]; }

    __device__ bool empty() const { return size_ == 0; }

  private:
    T data_[N];
    size_t size_;
};

class material;

class hit_record
{
  public:
    float t = 0.0f;
    point3 hit_point = point3();
    vec3 normal_vector = vec3();
    bool front_face = false;
    material *mat;

    __device__ inline void set_face_normal(const ray &r, const vec3 &outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0.0f;
        normal_vector = front_face ? outward_normal : -outward_normal;
    }
};

class hittable
{
  public:
    __device__ virtual ~hittable(){};
    __device__ virtual bool hit(const ray &r, const interval &ray_t, hit_record &rec,
                                StaticStack<hittable *, 16> *stack) const = 0;
    __device__ virtual aabb bounding_box() const = 0;
    bool is_bvh = false;
};

class hittable_list : public hittable
{
  public:
    hittable **objects;
    int obj_num;
    __device__ hittable_list(hittable **objects, int obj_num) : objects(objects), obj_num(obj_num) {};

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec,
                        StaticStack<hittable *, 16> *stack = nullptr) const override
    {
        hit_record tmp_rec;
        bool hit_anything = false;
        float closest_t = ray_t.max;

        for (int i = 0; i < obj_num; ++i)
        {
            if (objects[i]->hit(r, interval(ray_t.min, closest_t), tmp_rec, stack))
            {
                hit_anything = true;
                closest_t = tmp_rec.t;
                rec = tmp_rec;
            }
        }

        return hit_anything;
    }

    __device__ aabb bounding_box() const override { return bbox; }

  private:
    aabb bbox;
};

#endif
