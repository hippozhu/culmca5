
#include <Eigen/Dense>
using namespace Eigen;


/* ------------------------- SVM dataset class -------------------------------*/
class SVMData{
  public:
    SVMData(string);
    ~SVMData();
  
    int nfeat;
    int ninst;
	int nclass;
    Inst *inst;
	unsigned typecount[4];
    double *data, *data_col;
	short *label;
	
    void readData(const char*, bool, double *);
	double *getDataPoint(int);
	void getVector(int,VectorXd&);
	void calcEdistMatrix(double *, int *);
	void getTargetOffset(int *, int []);
	int getTargetSize(int []);
	
  private:
	int getLineNo(const char*);
	int getNFeat(const char*);
	int getNClass();
	int stringToInt(string);
	double stringToFloat(string);
	double edist(int, int);
};
