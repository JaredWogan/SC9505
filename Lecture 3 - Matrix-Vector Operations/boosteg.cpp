#include "boost/multi_array.hpp"
#include <iostream>
#include <cassert>

int main () {
  // Create a 3D array that is 3 x 4 x 2
  typedef boost::multi_array<double, 3> array_type;
  array_type A(boost::extents[3][4][2]);

  // Assign values to the elements
  typedef array_type::index index;
  int values = 0;
  for(index i = 0; i != 3; ++i)
    for(index j = 0; j != 4; ++j)
      for(index k = 0; k != 2; ++k)
	A[i][j][k] = values++;

  // Verify values
  int verify = 0;
  for(index i = 0; i != 3; ++i)
    for(index j = 0; j != 4; ++j)
      for(index k = 0; k != 2; ++k) {
	    assert(A[i][j][k] == verify++);
        std::cout << "A[" << i << "][" << j << "][" << k << "] = " << A[i][j][k] << std::endl;
      }
  return 0;
}
