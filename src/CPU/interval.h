#ifndef INTERVAL_H
#define INTERVAL_H

#include <limits>

/**
 * Define a utility class, interval.
 */

constexpr double infinity = std::numeric_limits<double>::infinity();

/**
 * Class representing a numerical interval with minimum and maximum bounds.
 * Provides functions to check containment and size of the interval.
 */
class interval
{
public:
    double min, max; ///< Minimum and maximum bounds of the interval.

    /**
     * Default constructor that initializes the interval to be empty.
     * The default empty interval has min set to +infinity and max set to -infinity.
     */
    interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    /**
     * Constructor that initializes the interval with given minimum and maximum values.
     *
     * @param min Minimum bound of the interval.
     * @param max Maximum bound of the interval.
     */
    interval(double min, double max) : min(min), max(max) {}

    /**
     * Get the size of the interval.
     *
     * @return The size (length) of the interval. If the interval is empty (min > max), returns 0.0.
     */
    double size() const
    {
        if (max >= min)
            return max - min;
        return 0.0;
    }

    /**
     * Check if a value is within the bounds of the interval (inclusive).
     *
     * @param x Value to check.
     * @return True if the value is within the interval, false otherwise.
     */
    bool contains(double x) const
    {
        return min <= x && x <= max;
    }

    /**
     * Check if a value is strictly within the bounds of the interval (exclusive).
     *
     * @param x Value to check.
     * @return True if the value is strictly within the interval, false otherwise.
     */
    bool surrounds(double x) const
    {
        return min < x && x < max;
    }

    /**
     * Static instance representing an empty interval.
     * The empty interval has min set to +infinity and max set to -infinity.
     */
    static const interval empty;

    /**
     * Static instance representing the universe interval.
     * The universe interval has min set to -infinity and max set to +infinity, covering all possible values.
     */
    static const interval universe;
};

// Define static members for empty and universe intervals.
const interval interval::empty = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);

#endif
