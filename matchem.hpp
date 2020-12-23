#ifndef MATCHEM_HPP
#define MATCHEM_HPP

#include "matchem_config.hpp"
#include "matchem_exception.hpp"
#include "matchem_kokkos.hpp"

#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace matchem {

namespace tests {
struct UnitWrap;
}

////////////////////////////////////////////////////////////////////////////////
class Matchem
////////////////////////////////////////////////////////////////////////////////
{
 public:

  // Types
  using TeamPolicy = Kokkos::TeamPolicy<>;
  using MemberType = typename TeamPolicy::member_type;

  template <typename DataType>
  using view = Kokkos::View<DataType, Kokkos::LayoutRight>;

  using sides_view_t = view<int**>;

  static constexpr int SIZE = MatchemConfig::SET_SIZE;

  /**
   * Constructor - sets up a "null" game state that is not playable.
   */
  Matchem(const MatchemConfig& config);

  /**
   * Destructor - cleans up memory
   */
  ~Matchem() = default;

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////// PRIMARY INTERFACE ///////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  /**
   * run - Run the simulation
   */
  void run();

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////// QUERIES /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  /**
   * get_config - returns a references to the game configuration
   */
  const MatchemConfig& get_config() const { return m_config; }

  /**
   * operator<< - produces a nice-looking output that should convey the state
   *              of the game.
   */
  std::ostream& operator<<(std::ostream& out) const;

 protected: // ================ PRIVATE INTERFACE ================================

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////// FORBIDDEN METHODS /////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  Matchem(const Matchem&) = delete;
  Matchem& operator=(const Matchem&) = delete;

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////// INTERNAL METHODS //////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // Run an indivual game of matching, returns how many rounds it took to finish
  KOKKOS_FUNCTION
  int run_indv(const int ws_idx);

  // Initialize an indivual game of matching
  KOKKOS_FUNCTION
  void init_indv(const int ws_idx);

  // Ask for number of correct matches
  KOKKOS_FUNCTION
  int get_num_matches(const int ws_idx);

  // Ask for truth of an individual match. Details of how to select a match for
  // truth query is done here.
  KOKKOS_FUNCTION
  void ask_truth(const int ws_idx); // extension point

  // Create the best guess you can.
  KOKKOS_FUNCTION
  void make_guess(const int ws_idx); // extension point

  //////////////////////////////////////////////////////////////////////////////
  ///////////////////////////// DATA MEMBERS ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // m_config - The configuration for this game
  MatchemConfig m_config;

  TeamPolicy m_policy;
  TeamUtils<> m_tu;

  sides_view_t m_game_state; // idx represents id of side1, value represents side2
  sides_view_t m_known_info; // idx represents id of side1, value represents bitmask of known info
  sides_view_t m_guess_state; // idx represents id of side1, value represents side2

  //////////////////////////////////////////////////////////////////////////////
  /////////////////////////////// FRIENDS //////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  friend struct matchem::tests::UnitWrap;
};

////////////////////////////////////////////////////////////////////////////////
///////////////////////// ASSCOCIATED OPERATIONS ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& out, const Matchem& m);

}

#endif
