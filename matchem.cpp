#include "matchem.hpp"
#include "matchem_exception.hpp"

#include <sstream>
#include <chrono>
#include <type_traits>

namespace matchem {

////////////////////////////////////////////////////////////////////////////////
Matchem::Matchem(const MatchemConfig& config) :
////////////////////////////////////////////////////////////////////////////////
  m_config(config),
  m_policy(ExeSpaceUtils<>::get_default_team_policy(m_config.num_runs())),
  m_tu(m_policy),
  m_game_state("m_game_state", m_tu.get_num_concurrent_teams(), SIZE),
#ifdef EXTRA_TRACKING
  m_full_info( "m_full_info",  m_tu.get_num_concurrent_teams(), SIZE, MAX_ROUNDS),
  m_round_info("m_round_info", m_tu.get_num_concurrent_teams(), MAX_ROUNDS),
  m_odds_info( "m_odds_info",  m_tu.get_num_concurrent_teams(), SIZE, SIZE),
#endif
  m_known_info("m_known_info", m_tu.get_num_concurrent_teams(), SIZE),
  m_guess_state("m_guess_state", m_tu.get_num_concurrent_teams(), SIZE)
{
  std::cout << "Running with " << m_tu.get_num_concurrent_teams() << " concurrent teams" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
void Matchem::run()
////////////////////////////////////////////////////////////////////////////////
{
  const auto start = std::chrono::steady_clock::now();
  int total_rounds = 0;
  Kokkos::parallel_reduce("Matchem::run", m_policy, KOKKOS_LAMBDA(const MemberType& team, int& rounds) {
    const int ws_idx = m_tu.get_workspace_idx(team);

    init_indv(ws_idx);
    rounds += run_indv(ws_idx);

    m_tu.release_workspace_idx(team, ws_idx);
  }, total_rounds);

  std::cout << static_cast<double>(total_rounds) / m_config.num_runs() << " avg rounds per game" << std::endl;

  const auto finish = std::chrono::steady_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  const double report_time = 1e-6*duration.count();
  std::cout << "Simulation took " << report_time << " seconds" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
int Matchem::run_indv(const int ws_idx)
////////////////////////////////////////////////////////////////////////////////
{
  auto my_state = matchem::subview(m_game_state, ws_idx);
  int rounds = 0;
  int matches = 0;

  do {
    ask_truth(ws_idx, rounds);

    make_guess(ws_idx, rounds);

    matches = get_num_matches(ws_idx);

    process_guess_result(ws_idx, rounds, matches);

    ++rounds;
  } while(matches < SIZE);

  return rounds;
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
void Matchem::init_indv(const int ws_idx)
////////////////////////////////////////////////////////////////////////////////
{
  auto my_state = matchem::subview(m_game_state, ws_idx);
  auto my_guess = matchem::subview(m_guess_state, ws_idx);
  auto my_info  = matchem::subview(m_known_info, ws_idx);

  for (int i = 0; i < SIZE; ++i) {
    my_state(i) = i;
    my_guess(i) = -1;
    my_info(i)  = 0;
  }

  std::random_shuffle(&my_state(0), &my_state(0) + SIZE);

#ifdef EXTRA_TRACKING
  auto my_full_info  = matchem::subview(m_full_info, ws_idx);
  auto my_round_info = matchem::subview(m_round_info, ws_idx);
  auto my_odds_info  = matchem::subview(m_odds_info, ws_idx);

  for (int i = 0; i < MAX_ROUNDS; ++i) {
    my_round_info(i) = -1;
  }

  for (int i = 0; i < SIZE; ++i) {
    for (int r = 0; r < MAX_ROUNDS; ++r) {
      my_full_info(i, r) = -1;
    }
    for (int j = 0; j < SIZE; ++j) {
      my_odds_info(i, j) = -1;
    }
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
int Matchem::get_num_matches(const int ws_idx) const
////////////////////////////////////////////////////////////////////////////////
{
  auto my_state = matchem::subview(m_game_state, ws_idx);
  auto my_guess = matchem::subview(m_guess_state, ws_idx);

  int result = 0;
  for (int i = 0; i < SIZE; ++i) {
    if (my_state(i) == my_guess(i)) {
      ++result;
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
Matchem::MatchState Matchem::get_state(const int ws_idx, const int side1, const int side2) const
////////////////////////////////////////////////////////////////////////////////
{
  static_assert(sizeof(int) == sizeof(int16_t)*2, "Incompatible int sizes");
  static_assert(SIZE <= 16, "SIZE too big to fit into int16_t bitmask");

  auto my_info  = matchem::subview(m_known_info, ws_idx);

  int16_t* pieces = reinterpret_cast<int16_t*>(&my_info(side1));
  const int16_t known_matches = pieces[0];
  const int16_t known_misses  = pieces[1];

  if (is_setb(known_matches, side2)) {
    assert(!is_setb(known_misses, side2));
    return YES_MATCH;
  }
  else if (is_setb(known_misses, side2)) {
    assert(!is_setb(known_matches, side2));
    return NO_MATCH;
  }
  else {
    return UNKNOWN_MATCH;
  }
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
void Matchem::set_state(const int ws_idx, const int side1, const int side2, const MatchState state)
////////////////////////////////////////////////////////////////////////////////
{
  auto my_info  = matchem::subview(m_known_info, ws_idx);

  int16_t* pieces = reinterpret_cast<int16_t*>(&my_info(side1));
  int16_t& known_matches = pieces[0];
  int16_t& known_misses  = pieces[1];

  assert(state != UNKNOWN_MATCH);

  assert(!is_setb(known_matches, side2));
  assert(!is_setb(known_misses, side2));

  if (state == YES_MATCH) {
    setb(known_matches, side2);
    // no other items can match to this j
    for (int i = 0; i < SIZE; ++i) {
      if (i != side1) {
        int16_t* other_pieces       = reinterpret_cast<int16_t*>(&my_info(i));
        int16_t& other_known_misses = other_pieces[1];

        setb(other_known_misses, side2);
      }
    }
  }
  else {
    assert(state == NO_MATCH);

    setb(known_misses, side2);
  }
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
bool Matchem::has_match(const int ws_idx, const int side1) const
////////////////////////////////////////////////////////////////////////////////
{
  auto my_info  = matchem::subview(m_known_info, ws_idx);

  int16_t* pieces = reinterpret_cast<int16_t*>(&my_info(side1));
  const int16_t known_matches = pieces[0];

  return known_matches != 0;
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
int Matchem::get_match(const int ws_idx, const int side1) const
////////////////////////////////////////////////////////////////////////////////
{
  auto my_info  = matchem::subview(m_known_info, ws_idx);

  int16_t* pieces = reinterpret_cast<int16_t*>(&my_info(side1));
  const int16_t known_matches = pieces[0];

  for (int j = 0; j < SIZE; ++j) {
    if (is_setb(known_matches, j)) {
      return j;
    }
  }

  assert(false); // no match
  return -1;
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
int Matchem::get_num_pot_matches(const int ws_idx, const int side1) const
////////////////////////////////////////////////////////////////////////////////
{
  auto my_info  = matchem::subview(m_known_info, ws_idx);

  int16_t* pieces = reinterpret_cast<int16_t*>(&my_info(side1));
  const int16_t known_misses = pieces[0];

  int result = 0;
  for (int j = 0; j < SIZE; ++j) {
    if (!is_setb(known_misses, j)) {
      ++result;
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
void Matchem::ask_truth(const int ws_idx, const int round)
////////////////////////////////////////////////////////////////////////////////
{
  auto my_state = matchem::subview(m_game_state, ws_idx);

  const auto query = get_best_truth_query(ws_idx, round);
  const int side1_idx(query.first), side2_idx(query.second);
  assert(side1_idx != -1 && side2_idx != -1);

  // make the ask!
  const bool is_match = my_state(side1_idx) == side2_idx;
  set_state(ws_idx, side1_idx, side2_idx, is_match ? YES_MATCH : NO_MATCH);

  process_ask_result(ws_idx, round, side1_idx, side2_idx, is_match);
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
std::pair<int, int> Matchem::get_best_truth_query(const int ws_idx, const int round) const
////////////////////////////////////////////////////////////////////////////////
{
#ifdef EXTRA_TRACKING
  if (round == 0) {
    // we know nothing, so any guess is fine
    return std::make_pair(0, 0);
  }
  else {
    auto my_odds = matchem::subview(m_odds_info, ws_idx);
    int best_side1_idx(-1), best_side2_idx(-1);
    double best_odds_yet = 0.0;
    for (int i = 0; i < SIZE; ++i) {
      for (int j = 0; j < SIZE; ++j) {
        if (get_state(ws_idx, i, j) == UNKNOWN_MATCH) {
          const double odds = my_odds(i, j);
          assert(odds >= 0.0 && odds <= 1.0);

          // For now, just select the match with the best odds of being a correct. We'd
          // learn the most by selecting the closest to 50/50
          if (odds > best_odds_yet) {
            best_side1_idx = i;
            best_side2_idx = j;
            best_odds_yet = odds;
          }
        }
      }
    }
    return std::make_pair(best_side1_idx, best_side2_idx);
  }
#else
  for (int i = 0; i < SIZE; ++i) {

    if (!has_match(ws_idx, i)) {
      // no match is known yet for this item
      for (int j = 0; j < SIZE; ++j) {
        if (get_state(ws_idx, i, j) == UNKNOWN_MATCH) {
          return std::make_pair(i, j);
        }
      }
    }
  }
#endif

  assert(false); // Unable to select a query?
  return std::make_pair(-1, -1);
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
void Matchem::process_ask_result(
  const int ws_idx, const int round, const int side1_idx, const int side2_idx, bool was_match)
////////////////////////////////////////////////////////////////////////////////
{
#ifdef EXTRA_TRACKING
  auto my_odds = matchem::subview(m_odds_info, ws_idx);

  if (round == 0) {
    for (int i = 0; i < SIZE; ++i) {
      for (int j = 0; j < SIZE; ++j) {
        if (i == side1_idx && j == side2_idx) {
          if (was_match) {
            my_odds(i, j) = 1.0;
          }
          else {
            my_odds(i, j) = 0.0;
          }
        }
        else if (i == side1_idx || j == side2_idx) {
          if (was_match) {
            my_odds(i, j) = 0.0;
          }
          else {
            my_odds(i, j) = 1.0 / (SIZE-1);
          }
        }
        else {
          if (was_match) {
            my_odds(i, j) = 1.0 / (SIZE-1);
          }
          else {
            my_odds(i, j) = 1.0 / SIZE;
          }
        }
      }
    }
  }
  else {
    // if was a match, odds of others matching to this need to be redistributed to others
    if (was_match) {
      for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
          if (i == side1_idx) {
            if (j == side2_idx) {
              my_odds(i, j) = 1.0;
            }
            else {
              my_odds(i, j) = 0.0;
            }
          }
          else {
            if (j == side2_idx) {
              zeroout_and_redistribute(ws_idx, i, j);
            }
          }
        }
      }
    }
    else {
      // TODO - we can do better here
      zeroout_and_redistribute(ws_idx, side1_idx, side2_idx);
    }
  }
#endif
}

#ifdef EXTRA_TRACKING
////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
void Matchem::zeroout_and_redistribute(const int ws_idx, const int side1, const int side2)
////////////////////////////////////////////////////////////////////////////////
{
  auto my_odds = matchem::subview(m_odds_info, ws_idx);

  const double before_odds = my_odds(side1, side2);
  my_odds(side1, side2) = 0.0;

  const int num_pot_matches = get_num_pot_matches(ws_idx, side1);

  for (int j = 0; j < SIZE; ++j) {
    if (j != side2) {
      my_odds(side1, j) += before_odds / num_pot_matches;
    }
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
void Matchem::make_guess(const int ws_idx, const int round)
////////////////////////////////////////////////////////////////////////////////
{
  auto my_guess = matchem::subview(m_guess_state, ws_idx);
  auto my_odds  = matchem::subview(m_odds_info, ws_idx);

  // clear previous guesses
  for (int i = 0; i < SIZE; ++i) {
    my_guess(i) = -1;
  }

  for (int i = 0; i < SIZE; ++i) {

    if (has_match(ws_idx, i)) {
      my_guess(i) = get_match(ws_idx, i);
    }
    else {
      // no match is known yet for this item
#ifdef EXTRA_TRACKING
      double best_odds_yet = 0.0;
      int best_j = -1;
      for (int j = 0; j < SIZE; ++j) {
        if (get_state(ws_idx, i, j) == UNKNOWN_MATCH) {
          const double curr_odds = my_odds(i, j);
          if (curr_odds > best_odds_yet) {
            best_odds_yet = curr_odds;
            best_j = j;
          }
        }
      }
      my_guess(i) = best_j;
#else
      // just pick the first possibility (very dumb).
      for (int j = 0; j < SIZE; ++j) {
        if (get_state(ws_idx, i, j) == UNKNOWN_MATCH) {
          my_guess(i) = j;
          break;
        }
      }
#endif
    }
  }

#ifndef NDEBUG
  for (int i = 0; i < SIZE; ++i) {
    assert(my_guess(i) != -1);
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
void Matchem::process_guess_result(const int ws_idx, const int round, const int matches)
////////////////////////////////////////////////////////////////////////////////
{
#ifdef EXTRA_TRACKING
  // TODO
#endif
}

////////////////////////////////////////////////////////////////////////////////
std::ostream& Matchem::operator<<(std::ostream& out) const
////////////////////////////////////////////////////////////////////////////////
{
  out << "TODO\n";
  return out;
}

////////////////////////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& out, const Matchem& m)
////////////////////////////////////////////////////////////////////////////////
{
  return m.operator<<(out);
}

}
