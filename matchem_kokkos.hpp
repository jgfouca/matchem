#ifndef MATCHEM_KOKKOS
#define MATCHEM_KOKKOS

// Funnel all Kokkos includes through this file.

#pragma GCC system_header

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace matchem {

template <typename ExeSpace = Kokkos::DefaultExecutionSpace>
struct ExeSpaceUtils
{
  using TeamPolicy = Kokkos::TeamPolicy<ExeSpace>;

  static TeamPolicy get_default_team_policy (int ni)
  {
    return TeamPolicy(ni, 1); // one thread per-team for now
  }
};

template <typename ExeSpace = Kokkos::DefaultExecutionSpace>
class TeamUtilsCommonBase
{
 protected:
  int _team_size, _num_teams, _max_threads, _league_size;

  template <typename TeamPolicy>
  TeamUtilsCommonBase(const TeamPolicy& policy)
  {
    _max_threads = ExeSpace::concurrency();
    const int team_size = policy.team_size();
    _num_teams = _max_threads / team_size;
    _team_size = _max_threads / _num_teams;
    _league_size = policy.league_size();

    // We will never run more teams than the policy needs
    _num_teams = _num_teams > _league_size ? _league_size : _num_teams;
  }

 public:

  // How many thread teams can run concurrently
  int get_num_concurrent_teams() const { return _num_teams; }

  // How many threads can run concurrently
  int get_max_concurrent_threads() const { return _max_threads; }

  // How many ws slots are there
  int get_num_ws_slots() const { return _num_teams; }

  /*
   * Of the C concurrently running teams, which "slot" is open
   * for the given team.
   */
  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION
  int get_workspace_idx(const MemberType& /*team_member*/) const
  { return 0; }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION
  void release_workspace_idx(const MemberType& /*team_member*/, int /*ws_idx*/) const
  { }
};

template <typename ExeSpace = Kokkos::DefaultExecutionSpace>
class TeamUtils : public TeamUtilsCommonBase<ExeSpace>
{
public:
  template <typename TeamPolicy>
  TeamUtils(const TeamPolicy& policy) :
    TeamUtilsCommonBase<ExeSpace>(policy)
  { }
};

#ifdef KOKKOS_ENABLE_OPENMP
template <>
class TeamUtils<Kokkos::OpenMP> : public TeamUtilsCommonBase<Kokkos::OpenMP>
{
public:
  template <typename TeamPolicy>
  TeamUtils(const TeamPolicy& policy) :
    TeamUtilsCommonBase<Kokkos::OpenMP>(policy)
  { }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION
  int get_workspace_idx(const MemberType& /*team_member*/) const
  { return omp_get_thread_num() / this->_team_size; }
};
#endif

// Turn a View's MemoryTraits (traits::memory_traits) into the equivalent
// unsigned int mask. This is an implementation detail for Unmanaged; see next.
template <typename View>
struct MemoryTraitsMask {
  enum : unsigned int {
    value = ((View::traits::memory_traits::is_random_access ? Kokkos::RandomAccess : 0) |
             (View::traits::memory_traits::is_atomic ? Kokkos::Atomic : 0) |
             (View::traits::memory_traits::is_restrict ? Kokkos::Restrict : 0) |
             (View::traits::memory_traits::is_aligned ? Kokkos::Aligned : 0) |
             (View::traits::memory_traits::is_unmanaged ? Kokkos::Unmanaged : 0))
      };
};

// Make the input View Unmanaged, whether or not it already is. One might
// imagine that View::unmanaged_type would provide this.
//   Use: Unmanged<ViewType>
template <typename View>
using Unmanaged =
  // Provide a full View type specification, augmented with Unmanaged.
  Kokkos::View<typename View::traits::scalar_array_type,
               typename View::traits::array_layout,
               typename View::traits::device_type,
               Kokkos::MemoryTraits<
                 // All the current values...
                 MemoryTraitsMask<View>::value |
                 // ... |ed with the one we want, whether or not it's
                 // already there.
                 Kokkos::Unmanaged> >;

// Get a 1d subview of the i-th dimension of a 2d view
template <typename T, typename ...Parms>
KOKKOS_FORCEINLINE_FUNCTION
Unmanaged<Kokkos::View<T*, Parms...> >
subview (const Kokkos::View<T**, Parms...>& v_in, const int i,
         typename std::enable_if<std::remove_reference<decltype(v_in)>::type::Rank == 2>::type* = nullptr) {
  return Unmanaged<Kokkos::View<T*, Parms...> >(
    &v_in.impl_map().reference(i, 0), v_in.extent(1));
}

// Get a 2d subview of the i-th dimension of a 3d view
template <typename T, typename ...Parms>
KOKKOS_FORCEINLINE_FUNCTION
Unmanaged<Kokkos::View<T**, Parms...> >
subview (const Kokkos::View<T***, Parms...>& v_in, const int i,
         typename std::enable_if<std::remove_reference<decltype(v_in)>::type::Rank == 3>::type* = nullptr) {
  return Unmanaged<Kokkos::View<T**, Parms...> >(
    &v_in.impl_map().reference(i, 0, 0), v_in.extent(1), v_in.extent(2));
}

}

#endif
