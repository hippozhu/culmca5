#include "mycommon.h"
#include "SVMData.h"
#include "Neighborhood.h"
#include "KernelNeighborhood.h"
#include "Solver.h"

double alpha, nu[4], mu;
int k[4], kk = 40, n_train, n_test, n_validate, n_feat;
unsigned *tcount;

int main(int argc, char** argv){

  string filestem(argv[1]);
  alpha = atof(argv[2]);
  mu = atof(argv[3]);
  k[0] = atoi(argv[4]);
  k[1] = atoi(argv[5]);
  k[2] = atoi(argv[6]);
  k[3] = atoi(argv[7]);
  nu[0] = atof(argv[8]);
  nu[1] = atof(argv[9]);
  nu[2] = atof(argv[10]);
  nu[3] = atof(argv[11]);
  SVMData data_train(filestem+".train.train");
  n_train = data_train.ninst;
  n_feat = data_train.nfeat;
  SVMData data_validate(filestem+".train.validate");
  n_validate = data_validate.ninst;
  tcount = data_train.typecount;
  SVMData data_test(filestem+".test");
  n_test = data_test.ninst;  
  
  cout << "data loaded!" << endl;
  cout << data_train.nfeat << "*" << data_train.ninst << endl;
  
  dataToSymbol(&mu, sizeof(double), 0, 0, "d_mu");
  dataToSymbol(nu, sizeof(double) * 4, 0, 0, "d_nu");
  dataToSymbol(&data_train.nfeat, sizeof(int), 0, 0, "nfeat");
  
  dataToSymbol(&data_train.ninst, sizeof(int), 0, 0, "ntrain");
  dataToSymbol(&data_test.ninst, sizeof(int), 0, 0, "ntest");
  dataToSymbol(&data_validate.ninst, sizeof(int), 0, 0, "nvalidate");
  dataToSymbol(data_train.label, sizeof(short) * data_train.ninst, 0, sizeof(short *), "label_train");
  dataToSymbol(data_test.label, sizeof(short) * data_test.ninst, 0, sizeof(short *), "label_test");
  dataToSymbol(data_validate.label, sizeof(short) * data_validate.ninst, 0, sizeof(short *), "label_validate");
  
  double *m = new double[data_train.ninst * data_train.ninst];
  int *ino = new int[data_train.ninst * data_train.ninst];
  data_train.calcEdistMatrix(m, ino);
  dataToSymbol(m, sizeof(double) * data_train.ninst * data_train.ninst, 0, sizeof(double *), "dist_knn");
  dataToSymbol(ino, sizeof(int) * data_train.ninst * data_train.ninst, 0, sizeof(int *), "ino_knn");
  delete []m;
  delete []ino;
  mallocToSymbol(sizeof(int) * data_train.ninst * kk, 0, sizeof(int *), "neighbor_knn");
  dataToSymbol(&kk, sizeof(int), 0, 0, "nnegibor");
  
  int t_size = data_train.getTargetSize(k);
  mallocToSymbol(sizeof(int) * t_size, 0, sizeof(int *), "target");
  
  int *t_offset = new int[data_train.ninst];
  data_train.getTargetOffset(t_offset, k);
  dataToSymbol(t_offset, sizeof(int) * data_train.ninst, 0, sizeof(int *), "target_offset");
  delete []t_offset;
  
  dataToSymbol(&data_train.nclass, sizeof(int), 0, 0, "nclass");
  dataToSymbol(k, sizeof(int) * 4, 0, 0, "nn");
    
  dataToSymbol(data_train.typecount, sizeof(int) * 4, 0, 0, "typecount");
  
  deviceInitInstList(data_train.inst, data_train.typecount, data_train.ninst);
  mallocToSymbol(sizeof(double) * t_size, 0, sizeof(double *), "dist_target");
  mallocToSymbol(sizeof(double) * data_train.typecount[TN] * data_train.typecount[FN], 0, sizeof(double *), "dist1");
  if (data_train.nclass == 4)
    mallocToSymbol(sizeof(double) * data_train.typecount[TP] * data_train.typecount[FP], 0, sizeof(double *), "dist2");
  
  deviceInitKernelMatrix(data_train.nfeat, data_train.ninst, data_test.ninst, data_validate.ninst, data_train.data_col, data_test.data_col, data_validate.data_col);
  
  MatrixXd O = MatrixXd::Random(data_train.nfeat, data_train.ninst);
  double *o = new double[data_train.nfeat * data_train.ninst];
  //for (int i = 0; i < O.size(); ++ i)
  //  o[i] = O(i/O.cols(), i%O.cols());
  for (int i = 0; i < data_train.nfeat * data_train.ninst; ++ i)
    o[i] = 2.0 * rand() / RAND_MAX - 1;
  dataToSymbol(o, sizeof(double) * data_train.nfeat * data_train.ninst, 0, sizeof(double *), "O");
  delete []o;
  mallocToSymbol(sizeof(double) * data_train.nfeat * data_train.ninst, sizeof(double *), sizeof(double *), "O");
  
  mallocToSymbol(sizeof(double) * data_train.ninst * data_train.ninst, 0, sizeof(double *), "t_target");
  mallocToSymbol(sizeof(double) * data_train.ninst * data_train.ninst, 0, sizeof(double *), "t_update");
  mallocToSymbol(sizeof(double) * data_train.ninst * data_train.ninst, 0, sizeof(double *), "t_triplet");
  
  kernelTest();
}

