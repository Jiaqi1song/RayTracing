#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "interval.h"
#include "aabb.h"

class material;
enum HittableType { SPHERE, QUAD, MEDIUM, TRIANGLE, HITTABLE_LIST, BVH, ROTATE, TRANSLATE };

class hit_record
{
public:
    float t = 0.0f;
    float u;
    float v;
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
    __device__ virtual ~hittable() {};
    __device__ virtual bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *state) const = 0;
    __device__ virtual aabb bounding_box() const = 0;
    __device__ virtual HittableType get_type() const = 0; 
    __device__ virtual float pdf_value(const point3& origin, const vec3& direction, curandState *state) const { return 0.0;}
    __device__ virtual vec3 random(const point3& origin, curandState *state) const {return vec3(1,0,0);}
    aabb bbox;
    bool is_bvh = false;
};

class hittable_list : public hittable
{
public:
    hittable **objects;
    int obj_num;

    __device__ hittable_list(hittable **objects, int obj_num) : objects(objects), obj_num(obj_num) 
    {   
        bbox = aabb::empty();
        for (int i = 0; i < obj_num; ++i)
        {
            bbox = aabb(bbox, objects[i]->bounding_box());
        }
    }

    __device__ HittableType get_type() const override { return HittableType::HITTABLE_LIST; }

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *state, StaticStack<T, N>& stack=nullptr) const override
    {
        hit_record tmp_rec;
        bool hit_anything = false;
        float closest_t = ray_t.max;

        for (int i = 0; i < obj_num; ++i)
        {
            
            if (objects[i]->hit(r, interval(ray_t.min, closest_t), tmp_rec, state))
            {
                hit_anything = true;
                closest_t = tmp_rec.t;
                rec = tmp_rec;
            }
        }

        return hit_anything;
    }

    __device__ float pdf_value(const point3& origin, const vec3& direction, curandState *state) const override {
        auto weight = 1.0 / obj_num;
        auto sum = 0.0;

        for (int i = 0; i < obj_num; ++i) {
            sum += weight * objects[i]->pdf_value(origin, direction, state);
        }

        return sum;
    }

    __device__ vec3 random(const point3& origin, curandState *state) const override {
        return objects[random_int(0, obj_num-1, state)]->random(origin, state);
    }

    __device__ aabb bounding_box() const override { return bbox; }

    aabb bbox;
};


class translate : public hittable {
  public:
    __device__ translate(hittable *object, const vec3& offset)
      : object(object), offset(offset)
    {
        bbox = object->bounding_box() + offset;
    }

    __device__ HittableType get_type() const override { return HittableType::TRANSLATE; }

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *state) const override {
        // Move the ray backwards by the offset
        ray offset_r(r.origin() - offset, r.direction());

        // Determine whether an intersection exists along the offset ray (and if so, where)
        if (!object->hit(offset_r, ray_t, rec, state))
            return false;

        // Move the intersection point forwards by the offset
        rec.hit_point += offset;

        return true;
    }

    __device__ aabb bounding_box() const override { return bbox; }
    aabb bbox;

  private:
    hittable *object;
    vec3 offset;
    
};


class rotate_y : public hittable {
  public:
    __device__ rotate_y(hittable *object, float angle) : object(object) {
        auto radians = degrees_to_radians(angle);
        sin_theta = sinf(radians);
        cos_theta = cosf(radians);
        bbox = object->bounding_box();

        point3 min( FLT_MAX,  FLT_MAX,  FLT_MAX);
        point3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    auto x = i*bbox.x.max + (1-i)*bbox.x.min;
                    auto y = j*bbox.y.max + (1-j)*bbox.y.min;
                    auto z = k*bbox.z.max + (1-k)*bbox.z.min;

                    auto newx =  cos_theta*x + sin_theta*z;
                    auto newz = -sin_theta*x + cos_theta*z;

                    vec3 tester(newx, y, newz);

                    for (int c = 0; c < 3; c++) {
                        min[c] = fminf(min[c], tester[c]);
                        max[c] = fmaxf(max[c], tester[c]);
                    }
                }
            }
        }

        bbox = aabb(min, max);
    }

    __device__ HittableType get_type() const override { return HittableType::ROTATE; }

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *state) const override {

        // Transform the ray from world space to object space.

        auto origin = point3(
            (cos_theta * r.origin().x()) - (sin_theta * r.origin().z()),
            r.origin().y(),
            (sin_theta * r.origin().x()) + (cos_theta * r.origin().z())
        );

        auto direction = vec3(
            (cos_theta * r.direction().x()) - (sin_theta * r.direction().z()),
            r.direction().y(),
            (sin_theta * r.direction().x()) + (cos_theta * r.direction().z())
        );

        ray rotated_r(origin, direction);

        // Determine whether an intersection exists in object space (and if so, where).

        if (!object->hit(rotated_r, ray_t, rec, state))
            return false;

        // Transform the intersection from object space back to world space.

        rec.hit_point = point3(
            (cos_theta * rec.hit_point.x()) + (sin_theta * rec.hit_point.z()),
            rec.hit_point.y(),
            (-sin_theta * rec.hit_point.x()) + (cos_theta * rec.hit_point.z())
        );

        rec.normal_vector = vec3(
            (cos_theta * rec.normal_vector.x()) + (sin_theta * rec.normal_vector.z()),
            rec.normal_vector.y(),
            (-sin_theta * rec.normal_vector.x()) + (cos_theta * rec.normal_vector.z())
        );

        return true;
    }

    __device__ aabb bounding_box() const override { return bbox; }
    aabb bbox;

  private:
    hittable *object;
    float sin_theta;
    float cos_theta;
    
};


#endif
