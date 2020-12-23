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
    ++rounds;

    ask_truth(ws_idx);

    make_guess(ws_idx);

    matches = get_num_matches(ws_idx);
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
    my_guess(i) = 0;
    my_info(i)  = 0;
  }

  std::random_shuffle(&my_state(0), &my_state(0) + SIZE);
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
int Matchem::get_num_matches(const int ws_idx)
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
void Matchem::ask_truth(const int ws_idx)
////////////////////////////////////////////////////////////////////////////////
{
  auto my_info  = matchem::subview(m_known_info, ws_idx);
  auto my_state = matchem::subview(m_game_state, ws_idx);

  static_assert(sizeof(int) == sizeof(int16_t)*2, "Incompatible int sizes");

  for (int i = 0; i < SIZE; ++i) {
    int16_t* pieces = reinterpret_cast<int16_t*>(&my_info(i));
    int16_t& known_matches = pieces[0];
    int16_t& known_misses  = pieces[1];

    if (known_matches == 0) {
      // no match is known yet for this item
      int j_to_ask = -1;
      for (int j = 0; j < SIZE; ++j) {
        if (!is_set(known_misses, j)) {
          j_to_ask = j;
          break;
        }
      }

      // make the ask!
      if (my_state(i) == j_to_ask) {
        // match!
        setb(known_matches, j_to_ask);

        // no other items can match to this j
        for (int k = 0; k < SIZE; ++k) {
          if (k != i) {
            int16_t* other_pieces       = reinterpret_cast<int16_t*>(&my_info(k));
            int16_t& other_known_misses = other_pieces[1];

            setb(other_known_misses, j_to_ask);
          }
        }
      }
      else {
        // no match
        setb(known_misses, j_to_ask);
      }

      return;
    }
  }

  assert(false); // No guess was able to be made
}

////////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
void Matchem::make_guess(const int ws_idx)
////////////////////////////////////////////////////////////////////////////////
{
  auto my_info  = matchem::subview(m_known_info, ws_idx);
  auto my_guess = matchem::subview(m_guess_state, ws_idx);

  // clear previous guesses
  for (int i = 0; i < SIZE; ++i) {
    my_guess(i) = -1;
  }

  for (int i = 0; i < SIZE; ++i) {
    int16_t* pieces = reinterpret_cast<int16_t*>(&my_info(i));
    int16_t& known_matches = pieces[0];
    int16_t& known_misses  = pieces[1];

    if (known_matches != 0) {
      // match is already known
      for (int j = 0; j < SIZE; ++j) {
        if (is_set(known_matches, j)) {
          my_guess(i) = j;
          break;
        }
      }
    }
    else {
      // no match is known yet for this item, just pick the first
      // possibility (very dumb).
      for (int j = 0; j < SIZE; ++j) {
        if (!is_set(known_misses, j)) {
          my_guess(i) = j;
          break;
        }
      }
    }
  }

#ifndef NDEBUG
  for (int i = 0; i < SIZE; ++i) {
    assert(my_guess(i) != -1);
  }
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
