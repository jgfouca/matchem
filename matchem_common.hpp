#ifndef MATCHEM_COMMON_HPP
#define MATCHEM_COMMON_HPP

#include <sstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

#include "matchem_kokkos.hpp"

/**
 * File contains some utility methods that are not associated with any
 * particular class.
 */

namespace matchem {

//method below should word for any object that defines << and all the primitives
template <class T>
std::string obj_to_str(const T& val)
{
  std::ostringstream ss;
  ss << val;
  return std::string(ss.str());
}

template <class A, class B>
std::ostream& operator<<(std::ostream& out, const std::pair<A, B>& p)
{
  out << "(" << p.first << ", " << p.second << ")";
  return out;
}

template <class T>
bool vector_contains(const std::vector<const T*>& vect, const T& item)
{
  for (unsigned i = 0; i < vect.size(); ++i) {
    if (*vect[i] == item) {
      return true;
    }
  }
  return false;
}

// bit operations
template <typename T>
KOKKOS_INLINE_FUNCTION
bool is_setb(const T val, const int bitidx)
{
  return ((val >> bitidx) & 1LL) == 1;
}

template <typename T>
KOKKOS_INLINE_FUNCTION
void setb(T& val, const int bitidx)
{
  val |= 1LL << bitidx;
}

template <typename T>
KOKKOS_INLINE_FUNCTION
void clearb(T& val, const int bitidx)
{
  val &= ~(1LL << bitidx);
}

template <int Size, typename ...Parms>
KOKKOS_FUNCTION
void check_even_spread(
  const Kokkos::View<int*, Parms...>& v_in,
  typename std::enable_if<std::remove_reference<decltype(v_in)>::type::Rank == 1>::type* = nullptr)
{
  int16_t counts = 0;
  for (int i = 0; i < Size; ++i) {
    const int value = v_in(i);
    assert(value >= 0 && value < Size);
    assert(!is_setb(counts, value));
    setb(counts, value);
  }
}

}

#endif
