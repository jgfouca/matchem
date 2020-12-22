#ifndef MATCHEM_FACADE_HPP
#define MATCHEM_FACADE_HPP

#include <string>

/**
 * This class act's as the outside world's only interface to this program.
 * In order words, entities outside this package would only need to make
 * calls to the methods below in order to use the program.
 *
 * Note: this class follows both the "singleton" and "facade" patterns.
 */

namespace matchem {

////////////////////////////////////////////////////////////////////////////////
class MatchemFacade
////////////////////////////////////////////////////////////////////////////////
{
 public:
  /**
   * instance - Returns the global MatchemFacade instance.
   *            (This class is a Singleton)
   */
  static const MatchemFacade& instance();

  /**
   * play - Runs the game according to the options encoded in the arguments.
   */
  void play(int argc, char** argv) const;

  /**
   * HELP - A string describing how to use this program
   */
  static const std::string HELP;

 private:

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////// FORBIDDEN METHODS /////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  MatchemFacade(const MatchemFacade&) = delete;
  MatchemFacade& operator=(const MatchemFacade&) = delete;

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////// INTERNAL METHODS //////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  /**
   * Constructor - Private because this is a singleton class.
   */
  MatchemFacade() {}
};

}

#endif
