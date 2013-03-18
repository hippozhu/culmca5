#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <float.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <iterator>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <set>

#define TN 0
#define TP 1
#define FP 2
#define FN 3

enum Type
{
    TRAIN,
    VALIDATE,
    TEST,
};

using namespace std;	
	
struct Inst{
  unsigned short ino;
  short label;
};

extern double alpha, nu[4], mu;
extern int k[4], kk, n_train, n_test, n_validate, n_feat;
extern unsigned *tcount;

extern void deviceInitKernelMatrix(int, int, int, int, double *, double *, double *);
extern void dataToSymbol(void *data, size_t data_size, size_t offset, size_t pointer_size, const char *deviceSymbol);
extern void mallocToSymbol(size_t data_size, size_t offset, size_t pointer_size, const char *deviceSymbol);
extern void deviceInitInstList(struct Inst *, unsigned *, unsigned);

extern void kernelTest();


//extern void deviceInitTarget(int *h_target, int trainninst, int, int *, int *, int *);
//extern void deviceInitMu(double, double[]);
//extern void deviceInitO(double *, int);
//extern void deviceInitTargetTerm(double *, int);
//extern void deviceInitUpdateTerm(int, int);
//extern void deviceInitTri(int);
//extern void deviceInitLabelTrain(struct Inst *, unsigned);
//extern void deviceInitLabelTest(struct Inst *, unsigned);