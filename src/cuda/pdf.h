#ifndef PDF_H
#define PDF_H

#include "hittable.h"
#include "onb.h"


__device__ float sphere_pdf_value(const vec3& direction, curandState *state) {
    return 1/ (4 * PI);
}

__device__ vec3 sphere_pdf_generate(curandState *state) {
    return random_unit_vector(state);
}


__device__ float cosine_pdf_value(const onb uvw, const vec3& direction, curandState *state) {
    auto cosine_theta = dot(unit_vector(direction), uvw.w());
    return fmaxf(0, cosine_theta/PI);
}

__device__ vec3 cosine_pdf_generate(const onb uvw, curandState *state) {
    return uvw.transform(random_cosine_direction(state));
}


__device__ float mixture_pdf_value(const float pdf_value1, const float pdf_value2) {
    return 0.5 * pdf_value1 + 0.5 * pdf_value2;
}

__device__ vec3 mixture_pdf_generate(const vec3 pdf1, const vec3 pdf2, curandState *state) {
    if (random_float(state) < 0.5)
        return pdf1;
    else
        return pdf2;
}

#endif