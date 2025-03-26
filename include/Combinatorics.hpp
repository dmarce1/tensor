/*
 * Combinatorics.hpp
 *
 *  Created on: Mar 26, 2025
 *      Author: dmarce1
 */

#ifndef INCLUDE_COMBINATORICS_HPP_
#define INCLUDE_COMBINATORICS_HPP_

#include <cstddef>

constexpr size_t factorial(size_t n) {
	if (n) {
		return n * factorial(n - 1);
	} else {
		return 1;
	}
}

#endif /* INCLUDE_COMBINATORICS_HPP_ */
