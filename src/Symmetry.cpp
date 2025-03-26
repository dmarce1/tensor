#include "Combinatorics.hpp"
#include "Permutation.hpp"
#include <algorithm>
#include <array>
#include <bitset>
#include <concepts>
#include <cstddef>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include <functional>

#include "IndexTuple.hpp"

#include "Utility.hpp"
#include <iostream>

template<size_t N>
struct IntegerPartition: public std::array<size_t, N> {
	using base_type = std::array<size_t, N>;
	size_t count(size_t i) const {
		std::plus<size_t> const op { };
		return std::accumulate(this->begin(), this->end(), 0, op);
	}
	bool valid() const {
		std::plus<size_t> const op { };
		return std::accumulate(this->begin(), this->end, 0, op) == N;
	}
	size_t size() const {
		size_t n = 0;
		size_t i = 0;
		while (n < N) {
			n += (*this)[i++];
		}
		return i;
	}
	bool operator<(IntegerPartition const &other) const {
		size_t const a = size();
		size_t const b = other.size();
		if (a < b) {
			return true;
		} else if (a > b) {
			return false;
		}
		for (size_t n = 0; n < N; n++) {
			if ((*this)[n] > other[n]) {
				return true;
			} else if ((*this)[n] < other[n]) {
				return false;
			}
		}
		return false;
	}
	IntegerPartition conjugate() const {
		IntegerPartition conj = { 0 };
		for (size_t n = 0; n < N; n++) {
			for (size_t m = 0; m < (*this)[n]; m++) {
				conj[m]++;
			}
		}
		return conj;
	}
};

template<size_t N>
struct YoungTableau {
	YoungTableau(IntegerPartition<N> const &P) {
		setPartition(P);
	}
	void setPartition(IntegerPartition<N> const &P) {
		lambda = P;
		std::iota(tableau.begin(), tableau.end(), 1);
		std::reverse(tableau.begin(), tableau.end());
	}
	size_t& operator()(size_t i, size_t j) {
		size_t const index = flatIndex(i, j);
		if (index != N) {
			return tableau[index];
		} else {
			static thread_local size_t zero;
			zero = 0;
			return zero;
		}
	}
	size_t& operator[](size_t i) {
		return lambda[i];
	}
	std::string toString() const {
		std::string str;
		size_t const numberWidth = std::log10(N) + 1;
		size_t const nRow = lambda.size();
		size_t flatIndex = 0;
		std::string number;
		str += "\n";
		size_t nCol = lambda[0];
		size_t width = nCol * (numberWidth + 1);
		for (size_t row = 0; row < nRow; row++) {
			str += "+";
			for (size_t i = 1; i < width; i++) {
				if (i % (numberWidth + 1)) {
					str += "-";
				} else {
					str += "+";
				}
			}
			str += "+\n|";
			nCol = lambda[row];
			width = nCol * (numberWidth + 1);
			for (size_t col = 0; col < nCol; col++) {
				number = std::to_string(tableau[flatIndex]);
				while (number.size() < numberWidth) {
					number = std::string(" ") + number;
				}
				str += number + "|";
				flatIndex++;
			}
			str += "\n";
		}
		str += "+";
		for (size_t i = 1; i < width; i++) {
			str += "-";
		}
		str += "+\n";
		return str;
	}
	size_t operator()(size_t i, size_t j) const {
		size_t const index = flatIndex(i, j);
		if (index != N) {
			return tableau[index];
		} else {
			return 0;
		}
	}
	size_t operator[](size_t i) const {
		return lambda[i];
	}
	YoungTableau conjugate() const {
		YoungTableau conj;
		conj.lambda = lambda.conjugate();
		for (size_t r = 0; r < conj.lambda.size(); r++) {
			for (size_t c = 0; c < conj.lambda[r]; c++) {
				conj(r, c) = this->operator()(c, r);
			}
		}
		return conj;
	}
	size_t hookLength(size_t row, size_t col) const {
		size_t hLength = lambda[row] - 1 - col;
		do {
			row++;
			hLength++;
		} while ((*this)(row, col) != 0);
		return hLength;
	}
	size_t dimIrrRepSn() const {
		size_t hookLengthProduct = 1;
		size_t const nRow = lambda.size();
		for (size_t row = 0; row < nRow; row++) {
			size_t const nCol = lambda[row];
			for (size_t col = 0; col < nCol; col++) {
				hookLengthProduct *= hookLength(row, col);
			}
		}
		return factorial(N) / hookLengthProduct;
	}
	size_t dimIrrRepGL(size_t dim) const {
		size_t hookLengthProduct = 1;
		size_t numerator = 1;
		size_t const nRow = lambda.size();
		for (size_t row = 0; row < nRow; row++) {
			size_t const nCol = lambda[row];
			for (size_t col = 0; col < nCol; col++) {
				hookLengthProduct *= hookLength(row, col);
				if (dim + col <= row) {
					numerator = 0;
					break;
				}
				numerator *= dim + col - row;
			}
		}
		return numerator / hookLengthProduct;
	}
private:
	size_t flatIndex(size_t i, size_t j) const {
		if (j >= lambda[i]) {
			return N;
		}
		size_t index = 0;
		size_t r = 0;
		while (i) {
			if (r >= lambda.size()) {
				return N;
			}
			index += lambda[r];
			i--;
			r++;
		}
		index += j;
		if (index >= N) {
			return N;
		}
		return index;
	}
	IntegerPartition<N> lambda;
	std::array<size_t, N> tableau;
};

template<size_t D, size_t R>
std::vector<IndexTuple<D, R>> generateTupleClass(Permutation<R> const &permutation, IndexTuple<D, R> const &prototype) {
	std::vector<IndexTuple<D, R>> results;
	IndexTuple<D, R> iTuple = prototype;
	do {
		results.emplace_back(iTuple);
		iTuple = permutation.apply(iTuple);
	} while (iTuple != prototype);
	return results;
}

template<size_t D, size_t R>
std::vector<std::vector<IndexTuple<D, R>>> generateTuples(Permutation<R> const &permutation) {
	std::vector<std::vector<IndexTuple<D, R>>> subgroups;
	using indices_type = IndexTuple<D, R>;
	std::bitset<IndexTuple<D, R>::elementCount()> visited;
	for (indices_type iTuple = indices_type::begin(); iTuple != indices_type::end(); iTuple++) {
		if (!visited[iTuple.flatIndex()]) {
			subgroups.push_back(generateTupleClass(permutation, iTuple));
			for (auto const &element : subgroups.back()) {
				visited[element.flatIndex()] = true;
			}
		}
	}
	return subgroups;
}

template<size_t N>
std::vector<IntegerPartition<N>> generateIntegerPartitions() {
	std::vector<IntegerPartition<N>> rc;
	std::function<void(IntegerPartition<N>&, size_t, size_t)> genParts;
	genParts = [&rc, &genParts](IntegerPartition<N> &tup, size_t sum, size_t digit) {
		if (sum == N) {
			auto tmp = tup;
			rc.push_back(tmp);
			return;
		}
		if (digit >= N) {
			return;
		}
		if (sum < N) {
			size_t end = (digit == 0) ? N : tup[digit - 1];
			for (size_t n = 1; n <= end; n++) {
				tup[digit] = n;
				genParts(tup, sum + n, digit + 1);
			}
			tup[digit] = 0;
		}
	};
	IntegerPartition<N> ipart;
	std::fill(ipart.begin(), ipart.end(), 1);
	genParts(ipart, 0, 0);
	std::sort(rc.begin(), rc.end());
	return rc;
}

template<size_t N>
std::vector<Permutation<N>> generateSymmetricGroup() {
	size_t const Nele = std::tgamma(N + 1);
	std::vector<Permutation<N>> group;
	group.resize(Nele);
	group[0] = Permutation<N>::identity;
	for (size_t n = 0; n < Nele - 1; n++) {
		group[n + 1] = group[n].next();
	}
	return group;
}

void testSymmetry() {
	constexpr size_t D = 4;
	constexpr size_t R = 4;
	auto parts = generateIntegerPartitions<R>();
	for (auto const &part : parts) {
		YoungTableau<R> young(part);
		std::cout << young.toString() << "dimIrrRepSn = " << std::to_string(young.dimIrrRepSn()) << "\n";
		std::cout << "dimIrrRepGL = " << std::to_string(young.dimIrrRepGL(D)) << "\n";
	}
}
