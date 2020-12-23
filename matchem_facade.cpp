#include "matchem_facade.hpp"
#include "matchem_config.hpp"
#include "matchem.hpp"

#include <cstdlib>
#include <ctime>
#include <fstream>

namespace matchem {

const std::string MatchemFacade::HELP =
  "matchem --mode=(basic|||) \n"
  "   First step: you must pick your mode. \n"
  "\n"
  "<config-options> \n"
  "   These options can be used for any of the modes, however the vast majority \n"
  "   of the time, you won't need to change these: \n"
  "\n"
  "   --srand=<random seed> \n"
  "       Choose the random seed. This can allow you to repeat test results etc. \n"
  "       Default means the 'time' function will be used to produce \n"
  "       a pseudo-random seed.\n"
  "   --num-runs=<number of simulations to run> \n"
  "       How many simulations to run, default is 1000 \n"
  "\n"
  "\n"
  "EXAMPLES: \n"
  "  Run test1 \n"
  "  % ./matchem --mode=basic \n";

////////////////////////////////////////////////////////////////////////////////
const MatchemFacade& MatchemFacade::instance()
////////////////////////////////////////////////////////////////////////////////
{
  static MatchemFacade sf;
  return sf;
}

////////////////////////////////////////////////////////////////////////////////
void MatchemFacade::play(int argc, char** argv) const
////////////////////////////////////////////////////////////////////////////////
{
  // defaults
  SimulationType sim_type  = BASIC;
  auto           rand_seed = std::time(0);
  int            num_runs  = 1000;

  //do the options parsing:
  if (argc == 1) {
    //if no args given, provide help
    std::cout << HELP << std::endl;
    return;
  }

  for (int opt_itr = 1; opt_itr < argc; ++opt_itr) {
    std::string opt = "", arg = "", full_arg = argv[opt_itr];

    //check for presence of a an arg that looks like a help request
    if (full_arg == "-h" || full_arg == "-help" || full_arg == "--help") {
      std::cout << HELP << std::endl;
      return;
    }

    //if necessary, split --opt=arg into opt, arg
    if (full_arg.find('=') != std::string::npos) {
      opt = full_arg.substr(0, full_arg.find('='));
      arg = full_arg.substr(full_arg.find('=') + 1);
    }
    else {
      opt = full_arg;
    }

    // process options
    if (opt == "--mode") {
      if (arg == "basic") {
        sim_type = BASIC;
      }
      else {
        std::cerr << "Unknown sim mode: " << arg << std::endl;
        return;
      }
    }
    else if (opt == "--srand") {
      rand_seed = std::atoi(arg.c_str());
    }
    else if (opt == "--num-runs") {
      num_runs = std::atoi(arg.c_str());
    }
    else {
      std::cerr << "Unknown option: " << opt << std::endl;
      return;
    }
  }

  srand(rand_seed);

  MatchemConfig config(sim_type, num_runs);

  std::cout << "Running simulation with config: " << std::endl;
  std::cout << config << std::endl;

  Matchem matchem(config);
  matchem.run();
}

}
