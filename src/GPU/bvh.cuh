#ifndef BVH_H
#define BVH_H

#include "aabb.h"
#include "hittable.h"

#include <algorithm>


class bvh_node : public hittable {
  public:
    __device__ bvh_node(hittable_list list) : bvh_node(list.objects, 0, list.objects.size()) {}

    __device__ bvh_node(std::vector<shared_ptr<hittable>>& objects, size_t start, size_t end) {
        // Build the bounding box of the span of source objects.
        bbox = aabb::empty;
        for (size_t object_index=start; object_index < end; object_index++)
            bbox = aabb(bbox, objects[object_index]->bounding_box());

        int axis = bbox.longest_axis();

        auto comparator = (axis == 0) ? box_x_compare
                        : (axis == 1) ? box_y_compare
                                      : box_z_compare;

        size_t object_span = end - start;

        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            left = objects[start];
            right = objects[start+1];
        } else {
            std::sort(std::begin(objects) + start, std::begin(objects) + end, comparator);

            auto mid = start + object_span/2;
            left = new bvh_node(objects, start, mid);
            right = new bvh_node(objects, mid, end);
        }
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        if (!bbox.hit(r, ray_t))
            return false;

        bool hit_left = left->hit(r, ray_t, rec);
        bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

        return hit_left || hit_right;
    }

    __device__ aabb bounding_box() const override { return bbox; }

  private:
    hittable *left;
    hittable *right;
    aabb bbox;

    __device__ static bool box_compare(
        const hittable *a, const hittable *b, int axis_index
    ) {
        auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
        auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
        return a_axis_interval.min < b_axis_interval.min;
    }

    __device__ static bool box_x_compare (const hittable *a, const hittable *b) {
        return box_compare(a, b, 0);
    }

    __device__ static bool box_y_compare (const hittable *a, const hittable *b) {
        return box_compare(a, b, 1);
    }

    __device__ static bool box_z_compare (const hittable *a, const hittable *b) {
        return box_compare(a, b, 2);
    }
};


#endif
