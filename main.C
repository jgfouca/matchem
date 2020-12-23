#include "matchem_facade.hpp"
#include "matchem_kokkos.hpp"

using namespace std;

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);

  matchem::MatchemFacade::instance().play(argc, argv);

  Kokkos::finalize();

  return 0;
}
