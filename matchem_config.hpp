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
                const int num_runs,
                const int set_size);

  SimulationType sim_type() const { return m_sim_type;}
  int num_runs() const { return m_num_runs; }
  int set_size() const { return m_set_size; }

  /**
   * operator<< - produces a nice-looking output that should convey the configuration
   */
  std::ostream& operator<<(std::ostream& out) const;

 private:

  SimulationType m_sim_type;
  int m_num_runs;
  int m_set_size;
};

std::ostream& operator<<(std::ostream& out, const MatchemConfig& config);

}

#endif
