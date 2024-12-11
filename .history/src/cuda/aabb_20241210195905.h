#ifndef AABB_H
#define AABB_H

class aabb 
{
  public:
    int flag = 2;
    interval x, y, z;

    __device__ aabb() {} 

    __device__ aabb(const interval& x, const interval& y, const interval& z) : x(x), y(y), z(z)
    {
        pad_to_minimums();
    }

    __device__ aabb(const point3& a, const point3& b) 
    {
        x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]);
        y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
        z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);
        pad_to_minimums();
    }

    __device__ aabb(const aabb& box0, const aabb& box1) 
    {
        x = interval(box0.x, box1.x);
        y = interval(box0.y, box1.y);
        z = interval(box0.z, box1.z);
    }

    __device__ const interval& axis_interval(int n) const 
    {
        if (n == 1) return y;
        if (n == 2) return z;
        return x;
    }

    __device__ bool hit(const ray& r, interval ray_t) const 
    {
        const point3& ray_orig = r.origin();
        const vec3&   ray_dir  = r.direction();

        for (int axis = 0; axis < 3; axis++) {
            const interval& ax = axis_interval(axis);
            const float adinv = 1.0 / ray_dir[axis];

            auto t0 = (ax.min - ray_orig[axis]) * adinv;
            auto t1 = (ax.max - ray_orig[axis]) * adinv;

            if (t0 < t1) {
                if (t0 > ray_t.min) ray_t.min = t0;
                if (t1 < ray_t.max) ray_t.max = t1;
            } else {
                if (t1 > ray_t.min) ray_t.min = t1;
                if (t0 < ray_t.max) ray_t.max = t0;
            }

            if (ray_t.max <= ray_t.min)
                return false;
        }
        return true;
    }

    __device__ int longest_axis() const 
    {
        // Returns the index of the longest axis of the bounding box.
        if (x.size() > y.size())
            return x.size() > z.size() ? 0 : 2;
        else
            return y.size() > z.size() ? 1 : 2;
    }

    __device__ static aabb empty() { return aabb(interval::empty(), interval::empty(), interval::empty()); }    
    __device__ static aabb universe() { return aabb(interval::universe(), interval::universe(), interval::universe()); }

  private:

    __device__ void pad_to_minimums() 
    {
        // Adjust the AABB so that no side is narrower than some delta, padding if necessary.

        float delta = 0.0001;
        if (x.size() < delta) x = x.expand(delta);
        if (y.size() < delta) y = y.expand(delta);
        if (z.size() < delta) z = z.expand(delta);
    }
};

__device__ aabb operator+(const aabb& bbox, const vec3& offset) 
{
    return aabb(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z());
}

__device__ aabb operator+(const vec3& offset, const aabb& bbox) 
{
    return bbox + offset;
}


#endif