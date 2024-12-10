#ifndef BVH_H
#define BVH_H

#include "aabb.h"
#include "hittable.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

struct BVH_Node {
    hittable* node;
    size_t start;
    size_t end;
};

// Dynamic stack for construction
template <typename T>
class DynamicStack {
public:
    __device__ DynamicStack(T* data, int capacity)
        : capacity_(capacity), size_(0) {
        data_ = data;
    }

    __device__ ~DynamicStack() {
        free(data_);
    }

    __device__ void push(const T& value) {
        if (size_ < capacity_) {
            data_[size_++] = value;
        }
    }

    __device__ T pop() {
        return data_[--size_];
    }

    __device__ bool empty() const {
        return size_ == 0;
    }

private:
    T* data_;
    int capacity_;
    int size_;
};

// Static stack for hit traversal
template <typename T, size_t N>
struct StaticStack {
    __device__ StaticStack() : size_(0) {}

    __device__ void push(const T& value) {
        data_[size_++] = value;
    }

    __device__ T pop() {
        return data_[--size_];
    }

    __device__ bool empty() const {
        return size_ == 0;
    }

private:
    T data_[N];
    size_t size_;
};

class bvh_node : public hittable {
public:
    __device__ bvh_node() : left(nullptr), right(nullptr) { is_bvh = true; }

    __device__ HittableType get_type() const override { return HittableType::BVH; }

    __device__ bool hit(const ray& r, const interval& ray_t, hit_record& rec, curandState* state, StaticStack<T, N>& stack) const override {
        
        // StaticStack<hittable*, 16> stack;
        hittable* root = (hittable*) this;
        stack.push(root);

        bool hit_anything = false;
        interval closest_so_far = ray_t;

        // Pre-order tree traversal
        while (!stack.empty()) {
            hittable* current = stack.pop();
            if (!current->bbox.hit(r, closest_so_far))
                continue;

            if (current->is_bvh) {
                bvh_node* node = (bvh_node*) current;
                stack.push(node->right);
                stack.push(node->left);
            } else {
                if (current->hit(r, closest_so_far, rec, state)) {
                    hit_anything = true;
                    closest_so_far.max = rec.t;
                }
            }
        }

        return hit_anything;
    }

    __device__ friend hittable* build_bvh_node(hittable** objects, BVH_Node *bvh_data, size_t object_count, curandState* state);

    __device__ aabb bounding_box() const override { return bbox; }

    aabb bbox;

private:
    hittable* left;
    hittable* right;
    
};

__device__ static bool box_compare(const hittable* a, const hittable* b, int axis_index) {
    auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
    auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
    return a_axis_interval.min < b_axis_interval.min;
}

__device__ static bool box_x_compare(const hittable* a, const hittable* b) {
    return box_compare(a, b, 0);
}

__device__ static bool box_y_compare(const hittable* a, const hittable* b) {
    return box_compare(a, b, 1);
}

__device__ static bool box_z_compare(const hittable* a, const hittable* b) {
    return box_compare(a, b, 2);
}

__device__ hittable* build_bvh_node(hittable** objects, BVH_Node *bvh_data, size_t object_count, curandState* state) {

    DynamicStack<BVH_Node> stack(bvh_data, 5000);
    
    bvh_node* root = new bvh_node();
    stack.push(BVH_Node{ root, 0, object_count });

    while (!stack.empty()) {
        BVH_Node current = stack.pop();
        bvh_node* current_node = (bvh_node*)current.node;

        size_t start = current.start;
        size_t end = current.end;
        size_t object_span = end - start;

        current_node->bbox = aabb::empty();
        for (size_t i = start; i < end; ++i) {
            current_node->bbox = aabb(current_node->bbox, objects[i]->bounding_box());
        }

        if (object_span == 1) {
            current_node->left = objects[start];
            current_node->right = objects[start];
        } else if (object_span == 2) {
            current_node->left = objects[start];
            current_node->right = objects[start + 1];
        } else {
            int axis = current_node->bbox.longest_axis();
            auto comparator = (axis == 0) ? box_x_compare
                            : (axis == 1) ? box_y_compare
                                          : box_z_compare;

            thrust::device_ptr<hittable*> dev_ptr(objects);
            thrust::sort(dev_ptr + start, dev_ptr + end, comparator);

            size_t mid = start + object_span / 2;
            bvh_node* left_child = new bvh_node();
            bvh_node* right_child = new bvh_node();

            current_node->left = left_child;
            current_node->right = right_child;

            stack.push(BVH_Node{ right_child, mid, end });
            stack.push(BVH_Node{ left_child, start, mid });
        }
    }

    return root;
}

#endif  
