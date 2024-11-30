#ifndef PERLIN_H
#define PERLIN_H

class perlin {
  public:
    __device__ perlin(curandState *state) {
        for (int i = 0; i < point_count; i++) {
            randvec[i] = unit_vector(vec3::random(state,-1,1));
        }

        perlin_generate_perm(perm_x,state);
        perlin_generate_perm(perm_y,state);
        perlin_generate_perm(perm_z,state);
    }

    __device__ float noise(const point3& p) const {
        auto u = p.x() - floorf(p.x());
        auto v = p.y() - floorf(p.y());
        auto w = p.z() - floorf(p.z());

        auto i = int(floorf(p.x()));
        auto j = int(floorf(p.y()));
        auto k = int(floorf(p.z()));
        vec3 c[2][2][2];

        for (int di=0; di < 2; di++)
            for (int dj=0; dj < 2; dj++)
                for (int dk=0; dk < 2; dk++)
                    c[di][dj][dk] = randvec[
                        perm_x[(i+di) & 255] ^
                        perm_y[(j+dj) & 255] ^
                        perm_z[(k+dk) & 255]
                    ];

        return perlin_interp(c, u, v, w);
    }

    __device__ float turb(const point3& p, int depth) const {
        auto accum = 0.0;
        auto temp_p = p;
        auto weight = 1.0;

        for (int i = 0; i < depth; i++) {
            accum += weight * noise(temp_p);
            weight *= 0.5;
            temp_p *= 2;
        }

        return fabsf(accum);
    }

  private:
    static const int point_count = 256;
    vec3 randvec[point_count];
    int perm_x[point_count];
    int perm_y[point_count];
    int perm_z[point_count];

    __device__ static void perlin_generate_perm(int* p, curandState *state) {
        for (int i = 0; i < point_count; i++)
            p[i] = i;

        permute(p, point_count, state);
    }

    __device__ static void permute(int* p, int n, curandState *state) {
        for (int i = n-1; i > 0; i--) {
            int target = random_int(0, i, state);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }

    __device__ static float perlin_interp(const vec3 c[2][2][2], float u, float v, float w) {
        auto uu = u*u*(3-2*u);
        auto vv = v*v*(3-2*v);
        auto ww = w*w*(3-2*w);
        auto accum = 0.0;

        for (int i=0; i < 2; i++)
            for (int j=0; j < 2; j++)
                for (int k=0; k < 2; k++) {
                    vec3 weight_v(u-i, v-j, w-k);
                    accum += (i*uu + (1-i)*(1-uu))
                           * (j*vv + (1-j)*(1-vv))
                           * (k*ww + (1-k)*(1-ww))
                           * dot(c[i][j][k], weight_v);
                }

        return accum;
    }
};


#endif