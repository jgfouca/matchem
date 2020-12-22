#include "matchem_facade.hpp"

using namespace std;

int main(int argc, char** argv)
{
  matchem::MatchemFacade::instance().play(argc, argv);

  return 0;
}
