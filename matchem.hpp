#ifndef MATCHEM_GAME_HPP
#define MATCHEM_GAME_HPP

#include "matchem_config.hpp"
#include "matchem_exception.hpp"

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
  /**
   * Constructor - sets up a "null" game state that is not playable.
   */
  Matchem(const MatchemConfig& config);

  /**
   * Destructor - cleans up memory
   */
  virtual ~Matchem() = default;

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

 private: // ================ PRIVATE INTERFACE ================================

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////// FORBIDDEN METHODS /////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  Matchem(const Matchem&) = delete;
  Matchem& operator=(const Matchem&) = delete;

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////// INTERNAL METHODS //////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////


  //////////////////////////////////////////////////////////////////////////////
  ///////////////////////////// DATA MEMBERS ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // m_config - The configuration for this game
  MatchemConfig m_config;

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
