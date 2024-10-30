#ifndef PDF_H
#define PDF_H

#include "hittable_list.cuh"
#include "onb.cuh"


class pdf {
  public:
    __device__ virtual ~pdf() {}

    __device__ virtual float value(const vec3& direction) const = 0;
    __device__ virtual vec3 generate() const = 0;
};


class sphere_pdf : public pdf {
  public:
    __device__ sphere_pdf() {}

    __device__ float value(const vec3& direction) const override {
        return 1/ (4 * pi);
    }

    __device__ vec3 generate() const override {
        return random_unit_vector();
    }
};


class cosine_pdf : public pdf {
  public:
    __device__ cosine_pdf(const vec3& w) : uvw(w) {}

    __device__ float value(const vec3& direction) const override {
        auto cosine_theta = dot(unit_vector(direction), uvw.w());
        return std::fmax(0, cosine_theta/pi);
    }

    __device__ vec3 generate() const override {
        return uvw.transform(random_cosine_direction());
    }

  private:
    onb uvw;
};


class hittable_pdf : public pdf {
  public:
    __device__ hittable_pdf(const hittable& objects, const point3& origin)
      : objects(objects), origin(origin)
    {}

    __device__ float value(const vec3& direction) const override {
        return objects.pdf_value(origin, direction);
    }

    __device__ vec3 generate() const override {
        return objects.random(origin);
    }

  private:
    const hittable& objects;
    point3 origin;
};


class mixture_pdf : public pdf {
  public:
    __device__ mixture_pdf(shared_ptr<pdf> p0, shared_ptr<pdf> p1) {
        p[0] = p0;
        p[1] = p1;
    }

    __device__ float value(const vec3& direction) const override {
        return 0.5 * p[0]->value(direction) + 0.5 *p[1]->value(direction);
    }

    __device__ vec3 generate() const override {
        if (random_float() < 0.5)
            return p[0]->generate();
        else
            return p[1]->generate();
    }

  private:
    shared_ptr<pdf> p[2];
};


#endif
