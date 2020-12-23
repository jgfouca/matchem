#ifndef matchem_config_hpp
#define matchem_config_hpp

#include <iostream>
#include <string>
#include <vector>

namespace matchem {

enum SimulationType {BASIC};

/**
 * This class encapsulates everything that is configurable in this program.
 */

////////////////////////////////////////////////////////////////////////////////
class MatchemConfig
////////////////////////////////////////////////////////////////////////////////
{
 public:

  MatchemConfig(const SimulationType sim_type,
                const int num_runs);

  SimulationType sim_type() const { return m_sim_type;}
  int num_runs() const { return m_num_runs; }

  /**
   * operator<< - produces a nice-looking output that should convey the configuration
   */
  std::ostream& operator<<(std::ostream& out) const;

  // want this to be a compile-time constant for now
  static constexpr int SET_SIZE = 10;

 private:

  SimulationType m_sim_type;
  int m_num_runs;
};

std::ostream& operator<<(std::ostream& out, const MatchemConfig& config);

}

#endif
