#ifndef ONB_H
#define ONB_H


class onb {
  public:
    __device__ onb(const vec3& n) {
        axis3 = unit_vector(n);
        vec3 a = (fabs(axis3.x()) > 0.9) ? vec3(0,1,0) : vec3(1,0,0);
        axis2 = unit_vector(cross(axis3, a));
        axis1 = cross(axis3, axis2);
    }

    __device__ const vec3& u() const { return axis1; }
    __device__ const vec3& v() const { return axis2; }
    __device__ const vec3& w() const { return axis3; }

    __device__ vec3 transform(const vec3& v) const {
        // Transform from basis coordinates to local space.
        return (v[0] * axis1) + (v[1] * axis2) + (v[2] * axis3);
    }

  private:
    vec3 axis1;
    vec3 axis2;
    vec3 axis3;
};


#endif