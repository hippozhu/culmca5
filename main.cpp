#include "mycommon.h"
#include "SVMData.h"
#include "Neighborhood.h"
#include "KernelNeighborhood.h"
#include "Solver.h"

int main(int argc, char** argv){

  int k[4];
  double nu[4];
  int NClass;
  string filestem(argv[1]);
  size_t found = filestem.find("2");
  if (found != std::string::npos)
    NClass = 2;
  else
    NClass = 4;
  double alpha = atof(argv[2]);
  double mu = atof(argv[3]);
  k[0] = atoi(argv[4]);
  k[1] = atoi(argv[5]);
  k[2] = atoi(argv[6]);
  k[3] = atoi(argv[7]);
  nu[0] = atof(argv[8]);
  nu[1] = atof(argv[9]);
  nu[2] = atof(argv[10]);
  nu[3] = atof(argv[11]);
  
  SVMData data_train(filestem+".train");
  SVMData data_test(filestem+".test");
  cout << "data loaded!" << endl;
  KernelNeighborhood nb(data_train, data_test, NClass, k);
  cout << nb.nfeat << "*" << nb.ninst << ", k=" << nb.k << endl;
  Solver s(nb, mu, alpha, nu);
  double *r = (double *)malloc(sizeof(double) * nb.nfeat * nb.ninst);
  kernelTest(nb.nfeat, nb.ninst, nb.ninst_test, k, r, mu, s.alpha, s.nu);
}

