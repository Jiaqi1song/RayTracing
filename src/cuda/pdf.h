#ifndef PDF_H
#define PDF_H

#include "hittable.h"
#include "onb.h"

class pdf {
  public:
    __device__ virtual ~pdf() {}

    __device__ virtual float value(const vec3& direction, curandState *state) const {return 0; }
    __device__ virtual vec3 generate(curandState *state) const {return vec3(0,0,0); }
};


class sphere_pdf : public pdf {
  public:
    __device__ sphere_pdf() {}

    __device__ float value(const vec3& direction, curandState *state) const override {
        return 1/ (4 * PI);
    }

    __device__ vec3 generate(curandState *state) const override {
        return random_unit_vector(state);
    }
};


class cosine_pdf : public pdf {
  public:
    __device__ cosine_pdf() {}
    __device__ cosine_pdf(const vec3& w) {
      uvw.build(w);
    }

    __device__ float value(const vec3& direction, curandState *state) const override {
        auto cosine_theta = dot(unit_vector(direction), uvw.w());
        return fmaxf(0, cosine_theta/PI);
    }

    __device__ vec3 generate(curandState *state) const override {
        return uvw.transform(random_cosine_direction(state));
    }

  private:
    onb uvw;
};


class hittable_pdf : public pdf {
  public:
    __device__ hittable_pdf() {}
    __device__ hittable_pdf(hittable **objects, const point3& origin)
      : objects(objects), origin(origin)
    {}

    __device__ float value(const vec3& direction, curandState *state) const override {
        return pdf_value(origin, direction, state);
    }

    __device__ vec3 generate(curandState *state) const override {
        return random(origin, state);
    }

    hittable **objects;
    int obj_num;
    point3 origin;
  
  private:
    __device__ float pdf_value(const point3& origin, const vec3& direction, curandState *state) const {
        auto weight = 1.0 / obj_num;
        auto sum = 0.0;

        for (int i = 0; i < obj_num; ++i) {
            sum += weight * objects[i]->pdf_value(origin, direction, state);
        }

        return sum;
    }

    __device__ vec3 random(const point3& origin, curandState *state) const {
        return objects[random_int(0, obj_num-1, state)]->random(origin, state);
    }

};


class mixture_pdf : public pdf {
  public:
    __device__ mixture_pdf() {}
    __device__ mixture_pdf(pdf *p0, pdf *p1) {
        p_0 = p0;
        p_1 = p1;
    }

    __device__ float value(const vec3& direction, curandState *state) const override {
        return 0.5 * p_0->value(direction, state) + 0.5 *p_1->value(direction, state);
    }

    __device__ vec3 generate(curandState *state) const override {
        if (random_float(state) < 0.5)
            return p_0->generate(state);
        else
            return p_1->generate(state);
    }

  private:
    pdf *p_0;
    pdf *p_1;
};

enum class pdf_type{
  UNKNOW,
  SPHERE,
  COSINE,
};

struct dynamic_pdf {
  __device__ dynamic_pdf(): type{pdf_type::UNKNOW}{}

  pdf_type type = pdf_type::UNKNOW;
  sphere_pdf sphere;
  cosine_pdf cosine;
  pdf empty_pdf;
};

#endif
