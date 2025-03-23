#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>


namespace Tensors {

/******************************************************************************/
/* Forward Declarations                                                       */
/******************************************************************************/


template<size_t, size_t, size_t...>
struct Antisymmetry;

template<char>
struct Index;

template<size_t, size_t, size_t...>
struct Symmetry;

template<typename, size_t, size_t, typename...>
struct Tensor;

template<typename, size_t, size_t, char...>
struct TensorExpression;


/******************************************************************************/
/* Helper Classes                                                             */
/******************************************************************************/

template<size_t I0, size_t I1, size_t...Is>
struct Antisymmetry {
   static constexpr std::array values = {I0, I1, Is...};
};

template<char C>
struct Index {
   static constexpr char value = C;
};

template<size_t I0, size_t I1, size_t...Is>
struct Symmetry {
   static constexpr std::array values = {I0, I1, Is...};
};


/******************************************************************************/
/* Tensor Declarations                                                        */
/******************************************************************************/

template<typename T, size_t D>
struct Tensor<T, D, 0> {
   constexpr T& operator[]();
   constexpr T const& operator[]() const;
private:
   static constexpr int computeIndex();
   static constexpr size_t Size = 1;
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 1> {
   constexpr T& operator[](size_t);
   template<char I> 
   constexpr auto operator()(Index<I>);
   constexpr T const& operator[](size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>) const;
private:
   static constexpr int computeIndex(size_t i);
   static constexpr size_t Size = D;
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 2, Antisymmetry<0, 1>> {
   constexpr T& operator[](size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>);
   constexpr T const& operator[](size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j);
   static constexpr size_t Size = (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 2, Symmetry<0, 1>> {
   constexpr T& operator[](size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>);
   constexpr T const& operator[](size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j);
   static constexpr size_t Size = (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 2> {
   constexpr T& operator[](size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>);
   constexpr T const& operator[](size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j);
   static constexpr size_t Size = (D * D);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 3, Antisymmetry<0, 1, 2>> {
   constexpr T& operator[](size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>);
   constexpr T const& operator[](size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k);
   static constexpr size_t Size = (D * std::max(D - 1, size_t(0)) * std::max(D - 2, size_t(0)) / 6);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 3, Symmetry<0, 1, 2>> {
   constexpr T& operator[](size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>);
   constexpr T const& operator[](size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k);
   static constexpr size_t Size = (D * (D + 1) * (D + 2) / 6);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 3, Antisymmetry<0, 2>> {
   constexpr T& operator[](size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>);
   constexpr T const& operator[](size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k);
   static constexpr size_t Size = D * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 3, Symmetry<0, 2>> {
   constexpr T& operator[](size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>);
   constexpr T const& operator[](size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k);
   static constexpr size_t Size = D * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 3, Antisymmetry<0, 1>> {
   constexpr T& operator[](size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>);
   constexpr T const& operator[](size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k);
   static constexpr size_t Size = D * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 3, Symmetry<0, 1>> {
   constexpr T& operator[](size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>);
   constexpr T const& operator[](size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k);
   static constexpr size_t Size = D * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 3, Antisymmetry<1, 2>> {
   constexpr T& operator[](size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>);
   constexpr T const& operator[](size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k);
   static constexpr size_t Size = D * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 3, Symmetry<1, 2>> {
   constexpr T& operator[](size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>);
   constexpr T const& operator[](size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k);
   static constexpr size_t Size = D * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 3> {
   constexpr T& operator[](size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>);
   constexpr T const& operator[](size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k);
   static constexpr size_t Size = (D * D * D);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * std::max(D - 1, size_t(0)) * std::max(D - 2, size_t(0)) * std::max(D - 3, size_t(0)) / 24);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * (D + 1) * (D + 2) * (D + 3) / 24);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<0, 2, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = D * (D * std::max(D - 1, size_t(0)) * std::max(D - 2, size_t(0)) / 6);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<0, 2, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = D * (D * (D + 1) * (D + 2) / 6);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<0, 1, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = D * (D * std::max(D - 1, size_t(0)) * std::max(D - 2, size_t(0)) / 6);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<0, 1, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = D * (D * (D + 1) * (D + 2) / 6);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * std::max(D - 1, size_t(0)) / 2) * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * (D + 1) / 2) * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * std::max(D - 1, size_t(0)) / 2) * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * (D + 1) / 2) * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<0, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * D) * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<0, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * D) * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<0, 1, 2>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = D * (D * std::max(D - 1, size_t(0)) * std::max(D - 2, size_t(0)) / 6);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<0, 1, 2>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = D * (D * (D + 1) * (D + 2) / 6);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * std::max(D - 1, size_t(0)) / 2) * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * (D + 1) / 2) * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * std::max(D - 1, size_t(0)) / 2) * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * (D + 1) / 2) * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * std::max(D - 1, size_t(0)) / 2) * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * (D + 1) / 2) * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * std::max(D - 1, size_t(0)) / 2) * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * (D + 1) / 2) * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<1, 2, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = D * (D * std::max(D - 1, size_t(0)) * std::max(D - 2, size_t(0)) / 6);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<1, 2, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = D * (D * (D + 1) * (D + 2) / 6);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<1, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * D) * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<1, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * D) * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<0, 2>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * D) * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<0, 2>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * D) * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<0, 1>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * D) * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<0, 1>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * D) * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<1, 2>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * D) * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<1, 2>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * D) * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Antisymmetry<2, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * D) * (D * std::max(D - 1, size_t(0)) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4, Symmetry<2, 3>> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * D) * (D * (D + 1) / 2);
   std::array<T, Size> V;
};

template<typename T, size_t D>
struct Tensor<T, D, 4> {
   constexpr T& operator[](size_t, size_t, size_t, size_t);
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t);
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t);
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t);
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t);
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t);
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t);
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t);
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>);
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>);
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>);
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>);
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>);
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>);
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>);
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>);
   constexpr T const& operator[](size_t, size_t, size_t, size_t) const;
   template<char I> 
   constexpr auto operator()(Index<I>, size_t, size_t, size_t) const;
   template<char J> 
   constexpr auto operator()(size_t, Index<J>, size_t, size_t) const;
   template<char I, char J> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, size_t) const;
   template<char K> 
   constexpr auto operator()(size_t, size_t, Index<K>, size_t) const;
   template<char I, char K> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, size_t) const;
   template<char J, char K> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, size_t) const;
   template<char I, char J, char K> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, size_t) const;
   template<char L> 
   constexpr auto operator()(size_t, size_t, size_t, Index<L>) const;
   template<char I, char L> 
   constexpr auto operator()(Index<I>, size_t, size_t, Index<L>) const;
   template<char J, char L> 
   constexpr auto operator()(size_t, Index<J>, size_t, Index<L>) const;
   template<char I, char J, char L> 
   constexpr auto operator()(Index<I>, Index<J>, size_t, Index<L>) const;
   template<char K, char L> 
   constexpr auto operator()(size_t, size_t, Index<K>, Index<L>) const;
   template<char I, char K, char L> 
   constexpr auto operator()(Index<I>, size_t, Index<K>, Index<L>) const;
   template<char J, char K, char L> 
   constexpr auto operator()(size_t, Index<J>, Index<K>, Index<L>) const;
   template<char I, char J, char K, char L> 
   constexpr auto operator()(Index<I>, Index<J>, Index<K>, Index<L>) const;
private:
   static constexpr int computeIndex(size_t i, size_t j, size_t k, size_t l);
   static constexpr size_t Size = (D * D * D * D);
   std::array<T, Size> V;
};

/******************************************************************************/
/* Expression Declarations                                                    */
/******************************************************************************/

template<typename T, size_t D>
struct TensorExpression<T, D, 0> {
   TensorExpression(T);
private:
   T handle;
};

template<typename T, size_t D, char I>
struct TensorExpression<T, D, 1, I> {
   TensorExpression(T);
private:
   T handle;
};

template<typename T, size_t D, char I, char J>
struct TensorExpression<T, D, 2, I, J> {
   TensorExpression(T);
private:
   T handle;
};

template<typename T, size_t D, char I, char J, char K>
struct TensorExpression<T, D, 3, I, J, K> {
   TensorExpression(T);
private:
   T handle;
};

template<typename T, size_t D, char I, char J, char K, char L>
struct TensorExpression<T, D, 4, I, J, K, L> {
   TensorExpression(T);
private:
   T handle;
};


/******************************************************************************/
/* Tensor Implementations                                                     */
/******************************************************************************/

template<typename T, size_t D>
constexpr T& Tensor<T, D, 0>::operator[]() {
   int const index = computeIndex();
   return V[index];
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 0>::operator[]() const {
   int const index = computeIndex();
   return V[index];
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 0>::computeIndex() {
   size_t index = 0;
   return index;
}


template<typename T, size_t D>
constexpr T& Tensor<T, D, 1>::operator[](size_t i) {
   int const index = computeIndex(i);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 1>::operator()(Index<I>) {
   return TensorExpression<Tensor<T, D, 1> const&, D, 1, I>([this](size_t i) {
      return this->operator()(i);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 1>::operator[](size_t i) const {
   int const index = computeIndex(i);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 1>::operator()(Index<I>) const {
   return TensorExpression<Tensor<T, D, 1> const&, D, 1, I>([this](size_t i) {
      return this->operator()(i);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 1>::computeIndex(size_t i) {
   size_t index = i;
   return index;
}


template<typename T, size_t D>
constexpr T& Tensor<T, D, 2, Antisymmetry<0, 1>>::operator[](size_t i, size_t j) {
   int const index = computeIndex(i, j);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 2, Antisymmetry<0, 1>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 2, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j) {
   return TensorExpression<Tensor<T, D, 2, Antisymmetry<0, 1>> const&, D, 2, I>([this, j](size_t i) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 2, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>) {
   return TensorExpression<Tensor<T, D, 2, Antisymmetry<0, 1>> const&, D, 2, J>([this, i](size_t j) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 2, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>) {
   return TensorExpression<Tensor<T, D, 2, Antisymmetry<0, 1>> const&, D, 2, I, J>([this](size_t i, size_t j) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 2, Antisymmetry<0, 1>>::operator[](size_t i, size_t j) const {
   int const index = computeIndex(i, j);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 2, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j) const {
   return TensorExpression<Tensor<T, D, 2, Antisymmetry<0, 1>> const&, D, 2, I>([this, j](size_t i) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 2, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>) const {
   return TensorExpression<Tensor<T, D, 2, Antisymmetry<0, 1>> const&, D, 2, J>([this, i](size_t j) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 2, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>) const {
   return TensorExpression<Tensor<T, D, 2, Antisymmetry<0, 1>> const&, D, 2, I, J>([this](size_t i, size_t j) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 2, Antisymmetry<0, 1>>::computeIndex(size_t i, size_t j) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = 0;
   if(i < j) {
      std::swap(i, j);
      sign = -sign;
   } else if( i == j ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) / 2;
   index += j;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 2, Symmetry<0, 1>>::operator[](size_t i, size_t j) {
   int const index = computeIndex(i, j);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 2, Symmetry<0, 1>>::operator()(Index<I>, size_t j) {
   return TensorExpression<Tensor<T, D, 2, Symmetry<0, 1>> const&, D, 2, I>([this, j](size_t i) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 2, Symmetry<0, 1>>::operator()(size_t i, Index<J>) {
   return TensorExpression<Tensor<T, D, 2, Symmetry<0, 1>> const&, D, 2, J>([this, i](size_t j) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 2, Symmetry<0, 1>>::operator()(Index<I>, Index<J>) {
   return TensorExpression<Tensor<T, D, 2, Symmetry<0, 1>> const&, D, 2, I, J>([this](size_t i, size_t j) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 2, Symmetry<0, 1>>::operator[](size_t i, size_t j) const {
   int const index = computeIndex(i, j);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 2, Symmetry<0, 1>>::operator()(Index<I>, size_t j) const {
   return TensorExpression<Tensor<T, D, 2, Symmetry<0, 1>> const&, D, 2, I>([this, j](size_t i) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 2, Symmetry<0, 1>>::operator()(size_t i, Index<J>) const {
   return TensorExpression<Tensor<T, D, 2, Symmetry<0, 1>> const&, D, 2, J>([this, i](size_t j) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 2, Symmetry<0, 1>>::operator()(Index<I>, Index<J>) const {
   return TensorExpression<Tensor<T, D, 2, Symmetry<0, 1>> const&, D, 2, I, J>([this](size_t i, size_t j) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 2, Symmetry<0, 1>>::computeIndex(size_t i, size_t j) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   size_t index = 0;
   if(i < j) {
      std::swap(i, j);
   }
   index *= size0;
   index += i * (i + 1) / 2;
   index += j;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 2>::operator[](size_t i, size_t j) {
   int const index = computeIndex(i, j);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 2>::operator()(Index<I>, size_t j) {
   return TensorExpression<Tensor<T, D, 2> const&, D, 2, I>([this, j](size_t i) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 2>::operator()(size_t i, Index<J>) {
   return TensorExpression<Tensor<T, D, 2> const&, D, 2, J>([this, i](size_t j) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 2>::operator()(Index<I>, Index<J>) {
   return TensorExpression<Tensor<T, D, 2> const&, D, 2, I, J>([this](size_t i, size_t j) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 2>::operator[](size_t i, size_t j) const {
   int const index = computeIndex(i, j);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 2>::operator()(Index<I>, size_t j) const {
   return TensorExpression<Tensor<T, D, 2> const&, D, 2, I>([this, j](size_t i) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 2>::operator()(size_t i, Index<J>) const {
   return TensorExpression<Tensor<T, D, 2> const&, D, 2, J>([this, i](size_t j) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 2>::operator()(Index<I>, Index<J>) const {
   return TensorExpression<Tensor<T, D, 2> const&, D, 2, I, J>([this](size_t i, size_t j) {
      return this->operator()(i, j);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 2>::computeIndex(size_t i, size_t j) {
   size_t index = D *  j + i;
   return index;
}


template<typename T, size_t D>
constexpr T& Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator[](size_t i, size_t j, size_t k) {
   int const index = computeIndex(i, j, k);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(Index<I>, size_t j, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(size_t i, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(size_t i, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(Index<I>, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(size_t i, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator[](size_t i, size_t j, size_t k) const {
   int const index = computeIndex(i, j, k);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(Index<I>, size_t j, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(size_t i, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(size_t i, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(Index<I>, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(size_t i, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1, 2>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 3, Antisymmetry<0, 1, 2>>::computeIndex(size_t i, size_t j, size_t k) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) * std::max(D - 2, size_t(0)) / 6;
   int sign = 1;
   size_t index = 0;
   if(i < j) {
      std::swap(i, j);
      sign = -sign;
   } else if( i == j ) {
      sign = 0;
   }
   if(i < k) {
      std::swap(i, k);
      sign = -sign;
   } else if( i == k ) {
      sign = 0;
   }
   if(j < k) {
      std::swap(j, k);
      sign = -sign;
   } else if( j == k ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) * (i - 2) / 6;
   index += j * (j - 1) / 2;
   index += k;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator[](size_t i, size_t j, size_t k) {
   int const index = computeIndex(i, j, k);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(Index<I>, size_t j, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(size_t i, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(size_t i, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(Index<I>, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(size_t i, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator[](size_t i, size_t j, size_t k) const {
   int const index = computeIndex(i, j, k);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(Index<I>, size_t j, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(size_t i, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(size_t i, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(Index<I>, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(size_t i, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1, 2>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 3, Symmetry<0, 1, 2>>::computeIndex(size_t i, size_t j, size_t k) {
   static constexpr size_t size0 = D * (D + 1) * (D + 2) / 6;
   size_t index = 0;
   if(i < j) {
      std::swap(i, j);
   }
   if(i < k) {
      std::swap(i, k);
   }
   if(j < k) {
      std::swap(j, k);
   }
   index *= size0;
   index += i * (i + 1) * (i + 2) / 6;
   index += j * (j + 1) / 2;
   index += k;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 3, Antisymmetry<0, 2>>::operator[](size_t i, size_t j, size_t k) {
   int const index = computeIndex(i, j, k);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 3, Antisymmetry<0, 2>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(Index<I>, size_t j, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(size_t i, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(Index<I>, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(size_t i, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(Index<I>, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(size_t i, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(Index<I>, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 3, Antisymmetry<0, 2>>::operator[](size_t i, size_t j, size_t k) const {
   int const index = computeIndex(i, j, k);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(Index<I>, size_t j, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(size_t i, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(Index<I>, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(size_t i, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(Index<I>, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(size_t i, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 2>>::operator()(Index<I>, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 2>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 3, Antisymmetry<0, 2>>::computeIndex(size_t i, size_t j, size_t k) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = j;
   if(i < k) {
      std::swap(i, k);
      sign = -sign;
   } else if( i == k ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) / 2;
   index += k;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 3, Symmetry<0, 2>>::operator[](size_t i, size_t j, size_t k) {
   int const index = computeIndex(i, j, k);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(Index<I>, size_t j, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(size_t i, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(Index<I>, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(size_t i, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(Index<I>, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(size_t i, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(Index<I>, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 3, Symmetry<0, 2>>::operator[](size_t i, size_t j, size_t k) const {
   int const index = computeIndex(i, j, k);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(Index<I>, size_t j, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(size_t i, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(Index<I>, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(size_t i, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(Index<I>, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(size_t i, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 2>>::operator()(Index<I>, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 2>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 3, Symmetry<0, 2>>::computeIndex(size_t i, size_t j, size_t k) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   size_t index = j;
   if(i < k) {
      std::swap(i, k);
   }
   index *= size0;
   index += i * (i + 1) / 2;
   index += k;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 3, Antisymmetry<0, 1>>::operator[](size_t i, size_t j, size_t k) {
   int const index = computeIndex(i, j, k);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 3, Antisymmetry<0, 1>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(size_t i, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 3, Antisymmetry<0, 1>>::operator[](size_t i, size_t j, size_t k) const {
   int const index = computeIndex(i, j, k);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(size_t i, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<0, 1>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 3, Antisymmetry<0, 1>>::computeIndex(size_t i, size_t j, size_t k) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = k;
   if(i < j) {
      std::swap(i, j);
      sign = -sign;
   } else if( i == j ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) / 2;
   index += j;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 3, Symmetry<0, 1>>::operator[](size_t i, size_t j, size_t k) {
   int const index = computeIndex(i, j, k);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(Index<I>, size_t j, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(size_t i, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(Index<I>, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(size_t i, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(Index<I>, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(size_t i, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(Index<I>, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 3, Symmetry<0, 1>>::operator[](size_t i, size_t j, size_t k) const {
   int const index = computeIndex(i, j, k);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(Index<I>, size_t j, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(size_t i, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(Index<I>, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(size_t i, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(Index<I>, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(size_t i, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<0, 1>>::operator()(Index<I>, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<0, 1>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 3, Symmetry<0, 1>>::computeIndex(size_t i, size_t j, size_t k) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   size_t index = k;
   if(i < j) {
      std::swap(i, j);
   }
   index *= size0;
   index += i * (i + 1) / 2;
   index += j;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 3, Antisymmetry<1, 2>>::operator[](size_t i, size_t j, size_t k) {
   int const index = computeIndex(i, j, k);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 3, Antisymmetry<1, 2>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 3, Antisymmetry<1, 2>>::operator[](size_t i, size_t j, size_t k) const {
   int const index = computeIndex(i, j, k);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Antisymmetry<1, 2>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 3, Antisymmetry<1, 2>>::computeIndex(size_t i, size_t j, size_t k) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = i;
   if(j < k) {
      std::swap(j, k);
      sign = -sign;
   } else if( j == k ) {
      sign = 0;
   }
   index *= size0;
   index += j * (j - 1) / 2;
   index += k;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 3, Symmetry<1, 2>>::operator[](size_t i, size_t j, size_t k) {
   int const index = computeIndex(i, j, k);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 3, Symmetry<1, 2>>::operator[](size_t i, size_t j, size_t k) const {
   int const index = computeIndex(i, j, k);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3, Symmetry<1, 2>> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 3, Symmetry<1, 2>>::computeIndex(size_t i, size_t j, size_t k) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   size_t index = i;
   if(j < k) {
      std::swap(j, k);
   }
   index *= size0;
   index += j * (j + 1) / 2;
   index += k;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 3>::operator[](size_t i, size_t j, size_t k) {
   int const index = computeIndex(i, j, k);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3>::operator()(Index<I>, size_t j, size_t k) {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3>::operator()(size_t i, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3>::operator()(Index<I>, Index<J>, size_t k) {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3>::operator()(size_t i, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3>::operator()(Index<I>, size_t j, Index<K>) {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3>::operator()(size_t i, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3>::operator()(Index<I>, Index<J>, Index<K>) {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 3>::operator[](size_t i, size_t j, size_t k) const {
   int const index = computeIndex(i, j, k);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 3>::operator()(Index<I>, size_t j, size_t k) const {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, I>([this, j, k](size_t i) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 3>::operator()(size_t i, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, J>([this, i, k](size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 3>::operator()(Index<I>, Index<J>, size_t k) const {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, I, J>([this, k](size_t i, size_t j) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 3>::operator()(size_t i, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, K>([this, i, j](size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 3>::operator()(Index<I>, size_t j, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, I, K>([this, j](size_t i, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 3>::operator()(size_t i, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, J, K>([this, i](size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 3>::operator()(Index<I>, Index<J>, Index<K>) const {
   return TensorExpression<Tensor<T, D, 3> const&, D, 3, I, J, K>([this](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 3>::computeIndex(size_t i, size_t j, size_t k) {
   size_t index = D * (D *  k + j) + i;
   return index;
}


template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<0, 1, 2, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) * std::max(D - 2, size_t(0)) * std::max(D - 3, size_t(0)) / 24;
   int sign = 1;
   size_t index = 0;
   if(i < j) {
      std::swap(i, j);
      sign = -sign;
   } else if( i == j ) {
      sign = 0;
   }
   if(i < k) {
      std::swap(i, k);
      sign = -sign;
   } else if( i == k ) {
      sign = 0;
   }
   if(i < l) {
      std::swap(i, l);
      sign = -sign;
   } else if( i == l ) {
      sign = 0;
   }
   if(j < k) {
      std::swap(j, k);
      sign = -sign;
   } else if( j == k ) {
      sign = 0;
   }
   if(j < l) {
      std::swap(j, l);
      sign = -sign;
   } else if( j == l ) {
      sign = 0;
   }
   if(k < l) {
      std::swap(k, l);
      sign = -sign;
   } else if( k == l ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) * (i - 2) * (i - 3) / 24;
   index += j * (j - 1) * (j - 2) / 6;
   index += k * (k - 1) / 2;
   index += l;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<0, 1, 2, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) * (D + 2) * (D + 3) / 24;
   size_t index = 0;
   if(i < j) {
      std::swap(i, j);
   }
   if(i < k) {
      std::swap(i, k);
   }
   if(i < l) {
      std::swap(i, l);
   }
   if(j < k) {
      std::swap(j, k);
   }
   if(j < l) {
      std::swap(j, l);
   }
   if(k < l) {
      std::swap(k, l);
   }
   index *= size0;
   index += i * (i + 1) * (i + 2) * (i + 3) / 24;
   index += j * (j + 1) * (j + 2) / 6;
   index += k * (k + 1) / 2;
   index += l;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<0, 2, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) * std::max(D - 2, size_t(0)) / 6;
   int sign = 1;
   size_t index = j;
   if(i < k) {
      std::swap(i, k);
      sign = -sign;
   } else if( i == k ) {
      sign = 0;
   }
   if(i < l) {
      std::swap(i, l);
      sign = -sign;
   } else if( i == l ) {
      sign = 0;
   }
   if(k < l) {
      std::swap(k, l);
      sign = -sign;
   } else if( k == l ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) * (i - 2) / 6;
   index += k * (k - 1) / 2;
   index += l;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<0, 2, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) * (D + 2) / 6;
   size_t index = j;
   if(i < k) {
      std::swap(i, k);
   }
   if(i < l) {
      std::swap(i, l);
   }
   if(k < l) {
      std::swap(k, l);
   }
   index *= size0;
   index += i * (i + 1) * (i + 2) / 6;
   index += k * (k + 1) / 2;
   index += l;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<0, 1, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) * std::max(D - 2, size_t(0)) / 6;
   int sign = 1;
   size_t index = k;
   if(i < j) {
      std::swap(i, j);
      sign = -sign;
   } else if( i == j ) {
      sign = 0;
   }
   if(i < l) {
      std::swap(i, l);
      sign = -sign;
   } else if( i == l ) {
      sign = 0;
   }
   if(j < l) {
      std::swap(j, l);
      sign = -sign;
   } else if( j == l ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) * (i - 2) / 6;
   index += j * (j - 1) / 2;
   index += l;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<0, 1, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) * (D + 2) / 6;
   size_t index = k;
   if(i < j) {
      std::swap(i, j);
   }
   if(i < l) {
      std::swap(i, l);
   }
   if(j < l) {
      std::swap(j, l);
   }
   index *= size0;
   index += i * (i + 1) * (i + 2) / 6;
   index += j * (j + 1) / 2;
   index += l;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<0, 3>, Antisymmetry<1, 2>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   static constexpr size_t size1 = size0 * D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = 0;
   if(i < l) {
      std::swap(i, l);
      sign = -sign;
   } else if( i == l ) {
      sign = 0;
   }
   if(j < k) {
      std::swap(j, k);
      sign = -sign;
   } else if( j == k ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) / 2;
   index += l;
   index *= size1;
   index += j * (j - 1) / 2;
   index += k;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<0, 3>, Antisymmetry<1, 2>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   static constexpr size_t size1 = size0 * D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = 0;
   if(i < l) {
      std::swap(i, l);
      sign = -sign;
   }
   if(j < k) {
      std::swap(j, k);
      sign = -sign;
   } else if( j == k ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i + 1) / 2;
   index += l;
   index *= size1;
   index += j * (j - 1) / 2;
   index += k;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<0, 3>, Symmetry<1, 2>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   static constexpr size_t size1 = size0 * D * (D + 1) / 2;
   int sign = 1;
   size_t index = 0;
   if(i < l) {
      std::swap(i, l);
      sign = -sign;
   } else if( i == l ) {
      sign = 0;
   }
   if(j < k) {
      std::swap(j, k);
      sign = -sign;
   }
   index *= size0;
   index += i * (i - 1) / 2;
   index += l;
   index *= size1;
   index += j * (j + 1) / 2;
   index += k;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<0, 3>, Symmetry<1, 2>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   static constexpr size_t size1 = size0 * D * (D + 1) / 2;
   size_t index = 0;
   if(i < l) {
      std::swap(i, l);
   }
   if(j < k) {
      std::swap(j, k);
   }
   index *= size0;
   index += i * (i + 1) / 2;
   index += l;
   index *= size1;
   index += j * (j + 1) / 2;
   index += k;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<0, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<0, 3>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<0, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<0, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = D *  k + j;
   if(i < l) {
      std::swap(i, l);
      sign = -sign;
   } else if( i == l ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) / 2;
   index += l;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<0, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<0, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<0, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   size_t index = D *  k + j;
   if(i < l) {
      std::swap(i, l);
   }
   index *= size0;
   index += i * (i + 1) / 2;
   index += l;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<0, 1, 2>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) * std::max(D - 2, size_t(0)) / 6;
   int sign = 1;
   size_t index = l;
   if(i < j) {
      std::swap(i, j);
      sign = -sign;
   } else if( i == j ) {
      sign = 0;
   }
   if(i < k) {
      std::swap(i, k);
      sign = -sign;
   } else if( i == k ) {
      sign = 0;
   }
   if(j < k) {
      std::swap(j, k);
      sign = -sign;
   } else if( j == k ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) * (i - 2) / 6;
   index += j * (j - 1) / 2;
   index += k;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<0, 1, 2>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) * (D + 2) / 6;
   size_t index = l;
   if(i < j) {
      std::swap(i, j);
   }
   if(i < k) {
      std::swap(i, k);
   }
   if(j < k) {
      std::swap(j, k);
   }
   index *= size0;
   index += i * (i + 1) * (i + 2) / 6;
   index += j * (j + 1) / 2;
   index += k;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<0, 2>, Antisymmetry<1, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   static constexpr size_t size1 = size0 * D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = 0;
   if(i < k) {
      std::swap(i, k);
      sign = -sign;
   } else if( i == k ) {
      sign = 0;
   }
   if(j < l) {
      std::swap(j, l);
      sign = -sign;
   } else if( j == l ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) / 2;
   index += k;
   index *= size1;
   index += j * (j - 1) / 2;
   index += l;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<0, 2>, Antisymmetry<1, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   static constexpr size_t size1 = size0 * D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = 0;
   if(i < k) {
      std::swap(i, k);
      sign = -sign;
   }
   if(j < l) {
      std::swap(j, l);
      sign = -sign;
   } else if( j == l ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i + 1) / 2;
   index += k;
   index *= size1;
   index += j * (j - 1) / 2;
   index += l;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<0, 2>, Symmetry<1, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   static constexpr size_t size1 = size0 * D * (D + 1) / 2;
   int sign = 1;
   size_t index = 0;
   if(i < k) {
      std::swap(i, k);
      sign = -sign;
   } else if( i == k ) {
      sign = 0;
   }
   if(j < l) {
      std::swap(j, l);
      sign = -sign;
   }
   index *= size0;
   index += i * (i - 1) / 2;
   index += k;
   index *= size1;
   index += j * (j + 1) / 2;
   index += l;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<0, 2>, Symmetry<1, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   static constexpr size_t size1 = size0 * D * (D + 1) / 2;
   size_t index = 0;
   if(i < k) {
      std::swap(i, k);
   }
   if(j < l) {
      std::swap(j, l);
   }
   index *= size0;
   index += i * (i + 1) / 2;
   index += k;
   index *= size1;
   index += j * (j + 1) / 2;
   index += l;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<0, 1>, Antisymmetry<2, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   static constexpr size_t size1 = size0 * D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = 0;
   if(i < j) {
      std::swap(i, j);
      sign = -sign;
   } else if( i == j ) {
      sign = 0;
   }
   if(k < l) {
      std::swap(k, l);
      sign = -sign;
   } else if( k == l ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) / 2;
   index += j;
   index *= size1;
   index += k * (k - 1) / 2;
   index += l;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<0, 1>, Antisymmetry<2, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   static constexpr size_t size1 = size0 * D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = 0;
   if(i < j) {
      std::swap(i, j);
      sign = -sign;
   }
   if(k < l) {
      std::swap(k, l);
      sign = -sign;
   } else if( k == l ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i + 1) / 2;
   index += j;
   index *= size1;
   index += k * (k - 1) / 2;
   index += l;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<0, 1>, Symmetry<2, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   static constexpr size_t size1 = size0 * D * (D + 1) / 2;
   int sign = 1;
   size_t index = 0;
   if(i < j) {
      std::swap(i, j);
      sign = -sign;
   } else if( i == j ) {
      sign = 0;
   }
   if(k < l) {
      std::swap(k, l);
      sign = -sign;
   }
   index *= size0;
   index += i * (i - 1) / 2;
   index += j;
   index *= size1;
   index += k * (k + 1) / 2;
   index += l;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<0, 1>, Symmetry<2, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   static constexpr size_t size1 = size0 * D * (D + 1) / 2;
   size_t index = 0;
   if(i < j) {
      std::swap(i, j);
   }
   if(k < l) {
      std::swap(k, l);
   }
   index *= size0;
   index += i * (i + 1) / 2;
   index += j;
   index *= size1;
   index += k * (k + 1) / 2;
   index += l;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<1, 2, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) * std::max(D - 2, size_t(0)) / 6;
   int sign = 1;
   size_t index = i;
   if(j < k) {
      std::swap(j, k);
      sign = -sign;
   } else if( j == k ) {
      sign = 0;
   }
   if(j < l) {
      std::swap(j, l);
      sign = -sign;
   } else if( j == l ) {
      sign = 0;
   }
   if(k < l) {
      std::swap(k, l);
      sign = -sign;
   } else if( k == l ) {
      sign = 0;
   }
   index *= size0;
   index += j * (j - 1) * (j - 2) / 6;
   index += k * (k - 1) / 2;
   index += l;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<1, 2, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) * (D + 2) / 6;
   size_t index = i;
   if(j < k) {
      std::swap(j, k);
   }
   if(j < l) {
      std::swap(j, l);
   }
   if(k < l) {
      std::swap(k, l);
   }
   index *= size0;
   index += j * (j + 1) * (j + 2) / 6;
   index += k * (k + 1) / 2;
   index += l;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<1, 3>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<1, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = D *  k + i;
   if(j < l) {
      std::swap(j, l);
      sign = -sign;
   } else if( j == l ) {
      sign = 0;
   }
   index *= size0;
   index += j * (j - 1) / 2;
   index += l;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<1, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<1, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   size_t index = D *  k + i;
   if(j < l) {
      std::swap(j, l);
   }
   index *= size0;
   index += j * (j + 1) / 2;
   index += l;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<0, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<0, 2>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<0, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<0, 2>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = D *  l + j;
   if(i < k) {
      std::swap(i, k);
      sign = -sign;
   } else if( i == k ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) / 2;
   index += k;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<0, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<0, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<0, 2>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   size_t index = D *  l + j;
   if(i < k) {
      std::swap(i, k);
   }
   index *= size0;
   index += i * (i + 1) / 2;
   index += k;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<0, 1>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<0, 1>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<0, 1>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<0, 1>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<0, 1>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<0, 1>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = D *  l + k;
   if(i < j) {
      std::swap(i, j);
      sign = -sign;
   } else if( i == j ) {
      sign = 0;
   }
   index *= size0;
   index += i * (i - 1) / 2;
   index += j;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<0, 1>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<0, 1>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<0, 1>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<0, 1>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<0, 1>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   size_t index = D *  l + k;
   if(i < j) {
      std::swap(i, j);
   }
   index *= size0;
   index += i * (i + 1) / 2;
   index += j;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<1, 2>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<1, 2>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = D *  l + i;
   if(j < k) {
      std::swap(j, k);
      sign = -sign;
   } else if( j == k ) {
      sign = 0;
   }
   index *= size0;
   index += j * (j - 1) / 2;
   index += k;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<1, 2>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<1, 2>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<1, 2>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<1, 2>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   size_t index = D *  l + i;
   if(j < k) {
      std::swap(j, k);
   }
   index *= size0;
   index += j * (j + 1) / 2;
   index += k;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Antisymmetry<2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      throw std::runtime_error("Exception in T Tensor<T, D, 4, Antisymmetry<2, 3>>::operator[](...).\n");
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Antisymmetry<2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   if( index > 0 ) {
      return V[index - 1];
   } else if( index < 0 ) {
      return -V[-(index + 1)];
   } else /*if( index == 0 )*/ {
      return T(0);
   }
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Antisymmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Antisymmetry<2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Antisymmetry<2, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * std::max(D - 1, size_t(0)) / 2;
   int sign = 1;
   size_t index = D *  j + i;
   if(k < l) {
      std::swap(k, l);
      sign = -sign;
   } else if( k == l ) {
      sign = 0;
   }
   index *= size0;
   index += k * (k - 1) / 2;
   index += l;
   index++;
   if( sign == +1 ) {
      return index;
   } else if( sign == -1 ) {
      return -index;
   } else /*if( sign == 0 )*/ {
      return 0;
   }
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4, Symmetry<2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4, Symmetry<2, 3>>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4, Symmetry<2, 3>>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4, Symmetry<2, 3>> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4, Symmetry<2, 3>>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   static constexpr size_t size0 = D * (D + 1) / 2;
   size_t index = D *  j + i;
   if(k < l) {
      std::swap(k, l);
   }
   index *= size0;
   index += k * (k + 1) / 2;
   index += l;
   return index;
}

template<typename T, size_t D>
constexpr T& Tensor<T, D, 4>::operator[](size_t i, size_t j, size_t k, size_t l) {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, size_t j, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, Index<J>, size_t k, size_t l) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, size_t j, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, Index<J>, Index<K>, size_t l) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, size_t j, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, Index<J>, size_t k, Index<L>) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, size_t j, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr T const& Tensor<T, D, 4>::operator[](size_t i, size_t j, size_t k, size_t l) const {
   int const index = computeIndex(i, j, k, l);
   return V[index];
}

template<typename T, size_t D>
template<char I> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, size_t j, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I>([this, j, k, l](size_t i) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, J>([this, i, k, l](size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, Index<J>, size_t k, size_t l) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, J>([this, k, l](size_t i, size_t j) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, K>([this, i, j, l](size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, size_t j, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, K>([this, j, l](size_t i, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, J, K>([this, i, l](size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, Index<J>, Index<K>, size_t l) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, J, K>([this, l](size_t i, size_t j, size_t k) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char L> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, L>([this, i, j, k](size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char L> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, size_t j, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, L>([this, j, k](size_t i, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char L> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, J, L>([this, i, k](size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char L> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, Index<J>, size_t k, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, J, L>([this, k](size_t i, size_t j, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char K, char L> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, K, L>([this, i, j](size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char K, char L> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, size_t j, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, K, L>([this, j](size_t i, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char J, char K, char L> 
constexpr auto Tensor<T, D, 4>::operator()(size_t i, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, J, K, L>([this, i](size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
template<char I, char J, char K, char L> 
constexpr auto Tensor<T, D, 4>::operator()(Index<I>, Index<J>, Index<K>, Index<L>) const {
   return TensorExpression<Tensor<T, D, 4> const&, D, 4, I, J, K, L>([this](size_t i, size_t j, size_t k, size_t l) {
      return this->operator()(i, j, k, l);
   });
}

template<typename T, size_t D>
constexpr int Tensor<T, D, 4>::computeIndex(size_t i, size_t j, size_t k, size_t l) {
   size_t index = D * (D * (D *  l + k) + j) + i;
   return index;
}


/******************************************************************************/
/* Expression Implementations                                                 */
/******************************************************************************/

template<typename T, size_t D>
TensorExpression<T, D, 0>::TensorExpression(T h) : handle(h) {
}

template<typename T, size_t D , char I>
TensorExpression<T, D, 1, I>::TensorExpression(T h) : handle(h) {
}

template<typename T, size_t D , char I, char J>
TensorExpression<T, D, 2, I, J>::TensorExpression(T h) : handle(h) {
}

template<typename T, size_t D , char I, char J, char K>
TensorExpression<T, D, 3, I, J, K>::TensorExpression(T h) : handle(h) {
}

template<typename T, size_t D , char I, char J, char K, char L>
TensorExpression<T, D, 4, I, J, K, L>::TensorExpression(T h) : handle(h) {
}


}

