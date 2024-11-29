#ifndef COLOR_H
#define COLOR_H

#include <algorithm>
#include "vec.h"

/**
 * Define color.
 *
 * Various operations are supported.
    - equal / not equal
    - color + color / color - color
    - color * color / color * double / double * color
    - color / double
    - lerp
 * All operations are inplaced supported.
 *
 * During calculation and creation,
 * the color doesn't require to be clamped into [0, 1],
 * but at the writing, the color is clamped.
 *
 * There are write color function for you.
 */

/**
 * Class representing a color with red, green, and blue components.
 * Provides various operations for manipulating colors.
 */
class color
{
protected:
    double _r, _g, _b; ///< Red, green, and blue components of the color.

public:
    /**
     * Default constructor that initializes color components to 0.
     */
    color() : _r(0.0), _g(0.0), _b(0.0) {}

    /**
     * Constructor that initializes color components to given values.
     *
     * @param r Red component.
     * @param g Green component.
     * @param b Blue component.
     */
    color(double r, double g, double b) : _r(r), _g(g), _b(b) {}

    /**
     * Getter for red component.
     *
     * @return The red component of the color.
     */
    double r() const { return _r; }

    /**
     * Getter for green component.
     *
     * @return The green component of the color.
     */
    double g() const { return _g; }

    /**
     * Getter for blue component.
     *
     * @return The blue component of the color.
     */
    double b() const { return _b; }

    /**
     * Clamp the color components to the range [0, 1].
     */
    void clamp()
    {
        _r = std::clamp(_r, 0.0, 1.0);
        _g = std::clamp(_g, 0.0, 1.0);
        _b = std::clamp(_b, 0.0, 1.0);
    }

    /**
     * Apply gamma correction to the color.
     *
     * @param gamma Gamma value for correction (default is 2.2).
     * @return A gamma-corrected color.
     */
    color gamma_correct(double gamma = 2.2)
    {
        return color(std::pow(_r, 1.0 / gamma),
                     std::pow(_g, 1.0 / gamma),
                     std::pow(_b, 1.0 / gamma));
    }

    /**
     * In-place addition of another color.
     *
     * @param v Color to add.
     * @return Reference to this color after addition.
     */
    color &operator+=(const color &v)
    {
        _r += v.r();
        _g += v.g();
        _b += v.b();
        return *this;
    }

    /**
     * In-place subtraction of another color.
     *
     * @param v Color to subtract.
     * @return Reference to this color after subtraction.
     */
    color &operator-=(const color &v)
    {
        _r -= v.r();
        _g -= v.g();
        _b -= v.b();
        return *this;
    }

    /**
     * In-place multiplication with another color.
     *
     * @param v Color to multiply.
     * @return Reference to this color after multiplication.
     */
    color &operator*=(const color &v)
    {
        _r *= v.r();
        _g *= v.g();
        _b *= v.b();
        return *this;
    }

    /**
     * In-place scalar multiplication.
     *
     * @param t Scalar value to multiply.
     * @return Reference to this color after scaling.
     */
    color &operator*=(double t)
    {
        _r *= t;
        _g *= t;
        _b *= t;
        return *this;
    }

    /**
     * In-place scalar division.
     *
     * @param t Scalar value to divide by.
     * @return Reference to this color after division.
     * @throws std::runtime_error if t is zero.
     */
    color &operator/=(double t)
    {
        if (t == 0)
            throw std::runtime_error("Division by zero");
        double inv_t = 1.0 / t;
        _r *= inv_t;
        _g *= inv_t;
        _b *= inv_t;
        return *this;
    }

    /**
     * Generate a random color with components in the range [0, 1).
     *
     * @return A random color.
     */
    static color random()
    {
        static std::random_device rd;                          // Seed for random number engine
        static std::mt19937 gen(rd());                         // Mersenne Twister random number generator
        static std::uniform_real_distribution<> dis(0.0, 1.0); // Distribution in [0, 1)

        return color(dis(gen), dis(gen), dis(gen));
    }

    /**
     * Generate a random color with components in the range [min, max).
     *
     * @return A random color.
     */
    static color random(double min, double max)
    {
        static std::random_device rd;                          // Seed for random number engine
        static std::mt19937 gen(rd());                         // Mersenne Twister random number generator
        static std::uniform_real_distribution<> dis(0.0, 1.0); // Distribution in [0, 1)

        double rand_r = min + (max - min) * dis(gen);
        double rand_g = min + (max - min) * dis(gen);
        double rand_b = min + (max - min) * dis(gen);
        return color(rand_r, rand_g, rand_b);
    }
};

/**
 * Output stream operator for color.
 *
 * @param out Output stream.
 * @param c Color to output.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &out, const color &c)
{
    // Translate the [0,1] component values to the byte range [0,255].
    int ir = int(255.999 * c.r());
    int ig = int(255.999 * c.g());
    int ib = int(255.999 * c.b());
    return out << "color(" << ir << ", " << ig << ", " << ib << ")";
}

/**
 * Equality operator for color.
 *
 * @param u First color.
 * @param v Second color.
 * @return True if the colors are equal, false otherwise.
 */
inline bool operator==(const color &u, const color &v)
{
    return std::fabs(u.r() - v.r()) < EPSILON &&
           std::fabs(u.g() - v.g()) < EPSILON &&
           std::fabs(u.b() - v.b()) < EPSILON;
}

/**
 * Inequality operator for color.
 *
 * @param u First color.
 * @param v Second color.
 * @return True if the colors are not equal, false otherwise.
 */
inline bool operator!=(const color &u, const color &v)
{
    return !(u == v);
}

/**
 * Color addition.
 *
 * @param u First color.
 * @param v Second color.
 * @return The result of adding the two colors.
 */
inline color operator+(const color &u, const color &v)
{
    return color(u.r() + v.r(), u.g() + v.g(), u.b() + v.b());
}

/**
 * Color subtraction.
 *
 * @param u First color.
 * @param v Second color.
 * @return The result of subtracting the second color from the first.
 */
inline color operator-(const color &u, const color &v)
{
    return color(u.r() - v.r(), u.g() - v.g(), u.b() - v.b());
}

/**
 * Color multiplication.
 *
 * @param u First color.
 * @param v Second color.
 * @return The result of multiplying the two colors.
 */
inline color operator*(const color &u, const color &v)
{
    return color(u.r() * v.r(), u.g() * v.g(), u.b() * v.b());
}

/**
 * Scalar multiplication of a color.
 *
 * @param u Color to scale.
 * @param t Scalar value.
 * @return The result of scaling the color by the scalar.
 */
inline color operator*(const color &u, double t)
{
    return color(u.r() * t, u.g() * t, u.b() * t);
}

/**
 * Scalar multiplication of a color (commutative).
 *
 * @param t Scalar value.
 * @param v Color to scale.
 * @return The result of scaling the color by the scalar.
 */
inline color operator*(double t, const color &v)
{
    return color(t * v.r(), t * v.g(), t * v.b());
}

/**
 * Scalar division of a color.
 *
 * @param u Color to divide.
 * @param t Scalar value.
 * @return The result of dividing the color by the scalar.
 * @throws std::runtime_error if t is zero.
 */
inline color operator/(const color &u, double t)
{
    if (t == 0)
        throw std::runtime_error("Division by zero");
    return color(u.r() / t, u.g() / t, u.b() / t);
}

/**
 * Linear interpolation between two colors.
 *
 * @param c1 First color.
 * @param c2 Second color.
 * @param t Interpolation factor (clamped between 0 and 1).
 * @return The interpolated color.
 */
inline color lerp(const color &c1, const color &c2, double t)
{
    t = std::clamp(t, 0.0, 1.0);
    return (1.0 - t) * c1 + t * c2;
}

/**
 * Write the color to an output stream with optional clamping and gamma correction.
 *
 * @param out Output stream.
 * @param c Color to write.
 */
void write_color(std::ostream &out, const color &c)
{
    // Translate the [0,1] component values to the byte range [0,255].
    // Default clamps the color and performs gamma correct.
    // The gamma correct is simplified to sqrt (when gamma == 2).
    auto translate = [](double x)
    {
        x = x > 0 ? std::sqrt(x) : 0;
        return int(255.999 * std::clamp(x, 0.0, 1.0));
    };
    // Write out the pixel color components.
    out << translate(c.r()) << " "
        << translate(c.g()) << " "
        << translate(c.b()) << "\n";
}

#endif
