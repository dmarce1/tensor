/*
 * Permutation.hpp
 *
 *  Created on: Mar 26, 2025
 *      Author: dmarce1
 */

#ifndef INCLUDE_PERMUTATION_HPP_
#define INCLUDE_PERMUTATION_HPP_

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

template<typename T>
concept RandomAccess = requires(T a, size_t i) {
	{a[i]}-> std::convertible_to<typename T::value_type>;
	{a.size()}-> std::convertible_to<size_t>;
};

template<size_t N>
struct Permutation: public std::array<size_t, N> {
	using base_type = std::array<size_t, N>;
	static constexpr Permutation identity = []() {
		Permutation p;
		std::iota(p.begin(), p.end(), 1);
		return p;
	}();
	static constexpr Permutation reversing = []() {
		Permutation p = identity;
		std::reverse(p.begin(), p.end());
		return p;
	}();
	Permutation inverse() const {
		base_type const &P = *this;
		Permutation I;
		for (size_t n = 0; n < N; n++) {
			I[P[n] - 1] = P[n];
		}
		return I;
	}
	template<RandomAccess Container>
	Container apply(Container const &B) const {
		base_type const &P = *this;
		Container A;
		for (size_t n = 0; n < N; n++) {
			A[P[n] - 1] = B[n];
		}
		return A;
	}
	Permutation operator*(Permutation const &B) const {
		base_type const &P = *this;
		Permutation A;
		for (size_t n = 0; n < N; n++) {
			A[P[n] - 1] = B[n];
		}
		return A;
	}
	Permutation operator/(Permutation const &B) const {
		return operator*(B.inverse());
	}
	Permutation& operator*=(Permutation const &B) {
		*this = *this * B;
		return *this;
	}
	Permutation& operator/=(Permutation const &B) {
		*this = *this / B;
		return *this;
	}
	size_t cycleLength() const {
		Permutation<N> Q = identity;
		size_t count = 0;
		do {
			Q = *this * Q;
			count++;
		} while (Q != identity);
		return count;
	}
	size_t inversionCount() const {
		base_type const &P = *this;
		size_t count = 0;
		for (size_t n = 0; n < N; n++) {
			for (size_t m = n + 1; m < N; m++) {
				if (P[n] > P[m]) {
					count++;
				}
			}
		}
		return count;
	}
	std::vector<Permutation> generateSubgroup() const {
		std::vector<Permutation> G;
		Permutation<N> Q = identity;
		do {
			G.emplace_back(Q);
			Q = apply(Q);
		} while (Q != identity);
		return G;
	}
	Permutation next() const {
		Permutation nextP = *this;
		if(!std::next_permutation(nextP.begin(), nextP.end())) {
			nextP = identity;
		}
		return nextP;
	}
	Permutation previous() const {
		Permutation prevP = *this;
		if(!std::prev_permutation(prevP.begin(), prevP.end())) {
			prevP = reversing;
		}
		return prevP;
	}
	int parity() const {
		return (inversionCount() & 1) ? -1 : +1;
	}
};




#endif /* INCLUDE_PERMUTATION_HPP_ */
