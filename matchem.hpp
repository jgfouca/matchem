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

// Configure optimizations. Keeping this compile-time for now to keep performance high
#define EXTRA_TRACKING

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

  using view_2d_int_t = view<int**>;
#ifdef EXTRA_TRACKING
  using view_3d_int_t = view<int***>;
  using view_3d_dbl_t = view<double***>;
#endif

  static constexpr int SIZE = MatchemConfig::SET_SIZE;

  static constexpr int MAX_ROUNDS = 64;

  enum MatchState {
    UNKNOWN_MATCH,
    NO_MATCH,
    YES_MATCH
  };

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

  //////////////////////////////// QUERIES /////////////////////////////////////

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

  ////////////////////////// GAME PHASES //////////////////////////////////

  // Run an indivual game of matching, returns how many rounds it took to finish
  KOKKOS_FUNCTION
  int run_indv(const int ws_idx);

  // Initialize an individual game of matching
  KOKKOS_FUNCTION
  void init_indv(const int ws_idx);

  // Ask for number of correct matches
  KOKKOS_FUNCTION
  int get_num_matches(const int ws_idx) const;

  // Ask for truth of an individual match.
  KOKKOS_FUNCTION
  void ask_truth(const int ws_idx, const int round);

  ////////////////////////// KNOWN INFO MGMT //////////////////////////////////

  // Ask for the state of a match
  KOKKOS_FUNCTION
  MatchState get_state(const int ws_idx, const int side1, const int side2) const;

  // Set the state of a match
  KOKKOS_FUNCTION
  void set_state(const int ws_idx, const int side1, const int side2, const MatchState state);

  // Does this side1 have a match yet?
  KOKKOS_FUNCTION
  bool has_match(const int ws_idx, const int side1) const;

  // Get match for side1
  KOKKOS_FUNCTION
  int get_match(const int ws_idx, const int side1) const;

  // Get first potential match for side1
  KOKKOS_FUNCTION
  int get_first_pot_match(const int ws_idx, const int side1) const;

    // Get first potential back match for side2
  KOKKOS_FUNCTION
  int get_first_pot_back_match(const int ws_idx, const int side2) const;

  // Get num potential matches for side1
  KOKKOS_FUNCTION
  int get_num_pot_matches(const int ws_idx, const int side1) const;

  // Get num potential back matches for side2
  KOKKOS_FUNCTION
  int get_num_pot_back_matches(const int ws_idx, const int side2) const;

  // Validate state
  void validate_state(const int ws_idx) const;

  ////////////////////////// EXTENSION POINTS //////////////////////////////////

  // Select most-useful truth query
  KOKKOS_FUNCTION
  std::pair<int, int> get_best_truth_query(const int ws_idx, const int round) const;

  // Process ask result
  KOKKOS_FUNCTION
  void process_ask_result(const int ws_idx, const int round, const int side1_idx, const int side2_idx, bool was_match);

  // Create the best guess you can.
  KOKKOS_FUNCTION
  void make_guess(const int ws_idx, const int round);

  // Process guess result
  KOKKOS_FUNCTION
  void process_guess_result(const int ws_idx, const int round, const int matches);

  //////////////////////////////////////////////////////////////////////////////
  ///////////////////////////// DATA MEMBERS ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // m_config - The configuration for this game
  MatchemConfig m_config;

  TeamPolicy m_policy;
  TeamUtils<> m_tu;

  // idx0 of all views is the ws_idx

  // this is secret, should only be accessed during initialization and truth queries
  view_2d_int_t m_game_state; // idx1 represents id of side1, value represents side2

#ifdef EXTRA_TRACKING
  // idx1  represents id of of side of side1
  // idx2  represents round
  // value represents id of of side2
  view_3d_int_t m_full_info;

  // idx1 represents the round, value represents num correct
  view_2d_int_t m_round_info;

  // idx1 represents id of side1, idx2 represents id of side2, value represents odds of match
  view_3d_dbl_t m_odds_info;
#endif
  view_2d_int_t m_known_info; // idx1 represents id of side1, value represents bitmask of known info

  view_2d_int_t m_guess_state; // idx1 represents id of side1, value represents side2

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
