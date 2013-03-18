#include "mycommon.h"
#include "SVMData.h"

SVMData::SVMData(string datafile){
  nfeat = getNFeat(datafile.c_str());
  ninst = getLineNo(datafile.c_str());
  inst = new Inst[ninst];
  label = new short[ninst];
  data = new double[nfeat*ninst];
  data_col = new double[nfeat*ninst];
  readData(datafile.c_str(), false, data);
  readData(datafile.c_str(), true, data_col);
  nclass = getNClass();
}

SVMData::~SVMData(){
  free(data);
  free(inst);
}

int SVMData::getLineNo(const char* filename){
  ifstream myfile(filename);
  if (!myfile.is_open()){
    cout << "Unable to open file: " << filename << endl;
	exit (EXIT_FAILURE);
  }
  myfile.unsetf(ios_base::skipws);
  int lineno = count(istream_iterator<char>(myfile), istream_iterator<char>(), '\n');
  myfile.close();
  return lineno;
}

int SVMData::getNFeat(const char* filename){
  ifstream myfile (filename);
  int nf = -1;
  string line;
  if (myfile.is_open()){
    getline(myfile,line);
    int pspace = line.find_first_of(" ");
	istringstream iss(line.substr(pspace+1));
      do{
        string sub;
        iss >> sub;
        if (sub.size()==0)
          break;
		int pcolon = sub.find_first_of(":");
		nf = stringToInt(sub.substr(0, pcolon));
      } while (iss); 
	  
  }
  else{
    cout << "Unable to open file: " << filename << endl;
	exit (EXIT_FAILURE);
  }
  return nf;
}

int SVMData::getNClass(){
  set<short> second (label, label + ninst);
  return second.size();
}

void SVMData::readData(const char* filename, bool columnwise, double *d){

  if (!columnwise){
    typecount[TN] = typecount[TP] = typecount[FP] = typecount[FN] = 0;
  }
  
  for (int i=0; i<ninst*nfeat; i++)
    d[i] = 0.0;
  string line;
  int lineno = 0;
  ifstream myfile (filename);
  if (myfile.is_open())
  {
    while ( myfile.good() )
    {
      getline (myfile,line);
      if (line.size()==0)
        break;
      int pspace = line.find_first_of(" ");
	  
	  if(!columnwise){
        inst[lineno].ino = lineno;
        inst[lineno].label = stringToInt(line.substr(0, pspace));
	    ++ typecount[inst[lineno].label];
		label[lineno] = inst[lineno].label;
	  }

      istringstream iss(line.substr(pspace+1));
      do{
        string sub;
        iss >> sub;
        if (sub.size()==0)
          break;
		int pcolon = sub.find_first_of(":");
		int idxFeat = stringToInt(sub.substr(0, pcolon))-1;
		double f = stringToFloat(sub.substr(pcolon+1));
		if (columnwise)
		  d[lineno + idxFeat*ninst] = f;
		else
		  d[lineno*nfeat + idxFeat] = f;
      } while (iss); 

      lineno ++;
    }
    myfile.close();
  }
  else{
    cout << "Unable to open file: " << filename << endl;
	exit (EXIT_FAILURE);
  }
}

double* SVMData::getDataPoint(int i){
  return data+i*nfeat;
}

void SVMData::getVector(int i, VectorXd& v){
  for(int j = 0; j < nfeat; j++)
    v(j) = (data+i*nfeat)[j];
}

int SVMData::stringToInt(string s){
  int result;
  stringstream ss(s);
  ss >> result;
  return result;
}

double SVMData::stringToFloat(string s){
  double result;
  stringstream ss(s);
  ss >> result;
  return result;
}

void SVMData::calcEdistMatrix(double *distMatrix, int *ino){
  for(int i = 0; i < ninst; i++)
    for(int j = 0; j < ninst; j++){
	  double d = DBL_MAX;
	  if (i != j && label[i] == label[j])
	      d = edist(i, j);
      distMatrix[i * ninst + j] = d;
	  ino[i * ninst + j] = j;
	}
}

double SVMData::edist(int i, int j){
  double *p1 = data + i * nfeat;
  double *p2 = data + j * nfeat;
  double dist = 0.0;
  for (int l = 0; l < nfeat; l++)
    dist += pow(p1[l]-p2[l], 2);
  return dist;
}

void SVMData::getTargetOffset(int *target_offset, int k[]){
  int base = 0;
  for(int i = 0; i < ninst; ++ i){  
    target_offset[i] = base;
	base += k[label[i]];
  }
}

int SVMData::getTargetSize(int k[]){
  int targetsize = 0;
  for(int i = 0; i < 4; ++ i)
    targetsize += typecount[i] * k[i];
  return targetsize;
}