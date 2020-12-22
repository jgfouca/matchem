#ifndef TESTS_COMMON_HPP
#define TESTS_COMMON_HPP

/*
 * Unit test infrastructure for matchem unit tests.
 *
 * Entities can friend matchem::tests::UnitWrap to give unit tests
 * access to private members.
 *
 * All unit test impls should be within an inner struct.
 */

namespace matchem {
namespace tests {

struct UnitWrap
{
  struct FullTests;
};

}
}

#endif
