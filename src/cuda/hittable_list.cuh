#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "aabb.cuh"
#include "hittable.cuh"

#include <vector>


class hittable_list : public hittable {
  public:
    std::vector<shared_ptr<hittable>> objects;

    __device__ hittable_list() {}
    __device__ hittable_list(shared_ptr<hittable> object) { add(object); }

    __device__ void clear() { objects.clear(); }

    __device__ void add(shared_ptr<hittable> object) {
        objects.push_back(object);
        bbox = aabb(bbox, object->bounding_box());
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (const auto& object : objects) {
            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    __device__ aabb bounding_box() const override { return bbox; }

    __device__ float pdf_value(const point3& origin, const vec3& direction) const override {
        auto weight = 1.0 / objects.size();
        auto sum = 0.0;

        for (const auto& object : objects)
            sum += weight * object->pdf_value(origin, direction);

        return sum;
    }

    __device__ vec3 random(const point3& origin) const override {
        auto int_size = int(objects.size());
        return objects[random_int(0, int_size-1)]->random(origin);
    }

  private:
    aabb bbox;
};


#endif
