#include "mycommon.h"

#define BSIZE 256

__constant__ int nfeat;
__constant__ int ntrain;
__constant__ int ntest;
__constant__ int nvalidate;
__constant__ int nclass;
__constant__ int nnegibor;
__constant__ double d_mu;
__constant__ double d_nu[4];
__constant__ int idx_o;

__constant__ int *target;
__constant__ double *km_train;
__constant__ double *km_test;
__constant__ double *km_validate;
__constant__ double *O[2];
__constant__ double *t_target;
__constant__ double *t_triplet;
__constant__ double *t_update;

__constant__ short *label_train, *label_test, *label_validate;
__constant__ struct Inst *type_inst[4];
__constant__ unsigned typecount[4];
__constant__ int *target_offset;
__constant__ int nn[4];

__constant__ double *dist_target;
__constant__ double *dist1;
__constant__ double *dist2;
__constant__ double *hinge_val;

__constant__ double *dist_knn;
__constant__ int *ino_knn;
__constant__ int *neighbor_knn;

__device__ double f_val;
__device__ double sub_fval[84];
__device__ double acc_knn;
__device__ int hits[4];

__device__ void kernelMatrix(double *km, double *d1, int n1, double *d2, int n2){
  int ub = n1 * n2;
  int stride = blockDim.x * gridDim.x;
  double c_val;
  int i, j;
  for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < ub; m += stride){
    i = m / n2;
	j = m % n2;
	c_val = .0;
	for (int n = 0; n < nfeat; ++ n)
	  c_val += pow(d1[n * n1 + i] - d2[n * n2 + j], 2);
	km[m] = exp(-c_val / nfeat);
  }
}

__global__ void calcKM(double *train, double *test, double *validate){
  kernelMatrix(km_train, train, ntrain, train, ntrain);
  kernelMatrix(km_test, test, ntest, train, ntrain);
  kernelMatrix(km_validate, validate, nvalidate, train, ntrain);
}

__device__ double getElement(double *m, int i, int j, int stride){
  return *(m + i * stride + j);
}

__device__ void setElement(double *m, int i, int j, int stride, double val){
  m[i * stride + j] = val;
}

__device__ int getElementInt(int *m, int i, int j, int stride){
  return *(m + i * stride + j);
}

__device__ void setElementInt(int *m, int i, int j, int stride, int val){
  m[i * stride + j] = val;
}

__device__ int getTargetByOffset(int ino, int kk){
  return target[target_offset[ino] + kk];
}

__device__ void setTargetByOffset(int ino, int kk, int t){
  target[target_offset[ino] + kk] = t;
}

__device__ int getTargetDist(int ino, int kk){
  return dist_target[target_offset[ino] + kk];
}

__device__ double calcDist(int i, double *km1, int j, double *km2){
  int tid = threadIdx.x;
  
  __shared__ double diff_k[256];
  __shared__ double sum[256];
  __shared__ double norm[64];
  
  if (tid < 64)
    norm[tid] = .0;
  
  int pos;
  for (int m = 0; m < (ntrain - 1)/blockDim.x + 1; ++ m){  
	__syncthreads();	
    pos = m * blockDim.x + tid;
	if (pos < ntrain)
      diff_k[tid] = getElement(km1, i, pos, ntrain) - getElement(km2, j, pos, ntrain);
	for (int d = 0; d < nfeat; ++ d){
	  __syncthreads();
	  if (pos < ntrain)
	    sum[tid] = getElement(O[idx_o], d, pos, ntrain) * diff_k[tid];
	  else
	    sum[tid] = .0;
		
	  int stride = blockDim.x/2;
	  while (stride > 0){
	    __syncthreads();
		if (tid < stride)
		  sum[tid] += sum[tid + stride];
		stride /= 2;
	  }
	  __syncthreads();
	  
	  if (tid == 0)
	    norm[d] += sum[0];
	}
  }
  
  if (tid < nfeat)
    norm[tid] = norm[tid]*norm[tid];
  
  __syncthreads();
  
  double s = .0;
  for (int d = 0; d < nfeat; ++ d)
	s += norm[d];
  return s;
}

__device__ void calcTargetDist(){
  int tid = threadIdx.x;
  int bid = blockIdx.x; 
  int i, j;
  if (tid == 0)
    sub_fval[bid] = .0;

  int c = 0;
  for (int m = 0; m < ntrain; ++ m)
    for (int n = 0; n < nn[label_train[m]]; ++ n){
	  i = m;
	  j = getTargetByOffset(m, n);
	  if(c%gridDim.x == bid){
	    double val = calcDist(i, km_train, j, km_train);
        if (tid == 0){
	      dist_target[target_offset[m] + n] = val;
          sub_fval[bid] += val;
	    }
	  }
	  ++ c;
	}
}

__device__ void updateDist(double *dist, struct Inst * inst1, int height, struct Inst * inst2, int width){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int i, j;
  for (int m = bid; m < height * width; m += gridDim.x){
    i = inst1[m / width].ino;
	j = inst2[m % width].ino;
	double val = calcDist(i, km_train, j, km_train);
    if (tid == 0)
	  dist[m] = val;
  }
}

__global__ void update2(){
  calcTargetDist();
  updateDist(dist1, type_inst[TN], typecount[TN], type_inst[FN], typecount[FN]);
  if (nclass == 4)
    updateDist(dist2, type_inst[TP], typecount[TP], type_inst[FP], typecount[FP]);
}

__device__ double hinge(double s){
  if (s <= -1.0)
    return .0;
  else if (s >= 0)
    return 1.0;
  else
    return 1 + s;
}

__device__ void updateTri(int idx1, int idx2, int idx3, double h){
  __syncthreads();
  for (int p = threadIdx.x; p < ntrain; p += blockDim.x)
    t_triplet[p * ntrain + idx1] += h * (getElement(km_train, idx2, p, ntrain) - getElement(km_train, idx3, p, ntrain));
}

__global__ void zeroT_triplet(){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size = gridDim.x * blockDim.x;
  for (int m = blockDim.x * bid + tid; m < ntrain * ntrain; m += size)
    t_triplet[m] = .0;  
}

__global__ void update3_2(){
  int bid = blockIdx.x;
  int i, j, l;
  double vdist, h;
  
  if (bid == 0 && threadIdx.x == 0)
    f_val = .0;
  
  for (int m = 0; m < typecount[TN] * typecount[FN]; ++ m){
    i = type_inst[TN][m / typecount[FN]].ino;
	l = type_inst[FN][m % typecount[FN]].ino;
    for (int kk = 0; kk < nn[TN]; ++ kk){
	  j = getTargetByOffset(i, kk);
	  vdist = 1 + dist_target[target_offset[i] + kk] - dist1[m];
	  if (vdist > 0 && blockIdx.x == 0 && threadIdx.x == 0)
	    f_val += vdist;
      h = hinge(vdist);
	  if (h > 0){
	    //if (label_train[i] == TP)
		h *= d_nu[label_train[i]];
	    if (i % gridDim.x == bid)
		  updateTri(i, l, j, h);
	    if (j % gridDim.x == bid)
		  updateTri(j, j, i, h);
	    if (l % gridDim.x == bid)
		  updateTri(l, i, l, h);
	  }
	}
	
    l = type_inst[TN][m / typecount[FN]].ino;
	i = type_inst[FN][m % typecount[FN]].ino;
    for (int kk = 0; kk < nn[FN]; ++ kk){
	  j = getTargetByOffset(i, kk);
	  vdist = 1 + dist_target[target_offset[i] + kk] - dist1[m];
	  if (vdist > 0 && blockIdx.x == 0 && threadIdx.x == 0)
	    f_val += vdist;
      h = hinge(vdist);
	  if (h > 0){
	    //if (label_train[i] == TP)
		h *= d_nu[label_train[i]];
	    if (i % gridDim.x == bid)
		  updateTri(i, l, j, h);
	    if (j % gridDim.x == bid)
		  updateTri(j, j, i, h);
	    if (l % gridDim.x == bid)
		  updateTri(l, i, l, h);
	  }
	}
  }
  
  if (nclass == 4){
  
  for (int m = 0; m < typecount[TP] * typecount[FP]; ++ m){
    i = type_inst[TP][m / typecount[FP]].ino;
	l = type_inst[FP][m % typecount[FP]].ino;
    for (int kk = 0; kk < nn[TP]; ++ kk){
	  j = getTargetByOffset(i, kk);
	  vdist = 1 + dist_target[target_offset[i] + kk] - dist2[m];
	  if (vdist > 0 && blockIdx.x == 0 && threadIdx.x == 0)
	    f_val += vdist;
      h = hinge(vdist);
	  if (h > 0){
		h *= d_nu[label_train[i]];
	    if (i % gridDim.x == bid)
		  updateTri(i, l, j, h);
	    if (j % gridDim.x == bid)
		  updateTri(j, j, i, h);
	    if (l % gridDim.x == bid)
		  updateTri(l, i, l, h);
	  }
	}
	
    l = type_inst[TP][m / typecount[FP]].ino;
	i = type_inst[FP][m % typecount[FP]].ino;
    for (int kk = 0; kk < nn[FP]; ++ kk){
	  j = getTargetByOffset(i, kk);
	  vdist = 1 + dist_target[target_offset[i] + kk] - dist2[m];
	  if (vdist > 0 && blockIdx.x == 0 && threadIdx.x == 0)
	    f_val += vdist;
      h = hinge(vdist);
	  if (h > 0){
		h *= d_nu[label_train[i]];
	    if (i % gridDim.x == bid)
		  updateTri(i, l, j, h);
	    if (j % gridDim.x == bid)
		  updateTri(j, j, i, h);
	    if (l % gridDim.x == bid)
		  updateTri(l, i, l, h);
	  }
	}
  }
  
  }
}

__global__ void calcFval(){ 
  if (blockIdx.x == 0 && threadIdx.x == 0)
	for (int i = 0; i < gridDim.x; ++ i)
	  f_val += sub_fval[i];
}

__global__ void updateUpdateTerm(double alpha){
  int size = gridDim.x * blockDim.x;
  for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < ntrain * ntrain; m += size){
    if (m/ntrain == m%ntrain)
      t_update[m] = 1 - 2 * alpha * (t_target[m] + d_mu * t_triplet[m]);
	else
      t_update[m] = - 2 * alpha * (t_target[m] + d_mu * t_triplet[m]);
  }
}

__global__ void copyO(){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size = gridDim.x * blockDim.x;
  for (int m = blockDim.x * bid + tid; m < nfeat * ntrain; m += size)
    O[idx_o][m] = O[1 - idx_o][m];
}

__global__ void zeroO(){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size = gridDim.x * blockDim.x;
  for (int m = blockDim.x * bid + tid; m < nfeat * ntrain; m += size)
    O[1 - idx_o][m] = .0; 
}

__global__ void updateO1(){
  int tid = threadIdx.x;
  int bid_row = blockIdx.x;
  int bid_col = blockIdx.y;
  int workingtid = min(BSIZE, ntrain - bid_col * BSIZE);
  
  if (tid < workingtid)
    O[1 - idx_o][bid_row * ntrain + bid_col * BSIZE + tid] = .0;
  
  
  for (int start = 0; start < ntrain; start += BSIZE){
	int len = min(BSIZE, ntrain - start);	
    for (int i = 0; i < len; ++ i){
	if (tid < workingtid){
	  double val = getElement(O[idx_o], bid_row, start + i, ntrain) * getElement(t_update, i + start, bid_col * BSIZE + tid, ntrain);
	  O[1 - idx_o][bid_row * ntrain + bid_col * BSIZE + tid] += val;
	}
	}
  }
}

__global__ void knnUpdateDist(){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size = gridDim.x;
	
  for(int m = bid; m < ntest * ntrain; m += size){
    int i = m / ntrain;
	int j = m % ntrain;
	
	double d = DBL_MAX;
	if (nclass == 2)
	  d = calcDist(i, km_test, j, km_train);
	else{
	  if (label_test[i] == label_train[j] || label_test[i] + label_train[j] == 3)
	    d = calcDist(i, km_test, j, km_train);
	}
	
	if (tid == 0){
	  ino_knn[m] = j;
      dist_knn[m] = d;
	}
  }
}

// lauched with # block = ntest
__global__ void knnFindNeighbor(){
  
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int len = ntrain / BSIZE;
  int start = tid * len;
  if (tid < ntrain % BSIZE){
    start += tid;
	++ len;
  }
  else
    start += ntrain % BSIZE;
  
  __syncthreads();
  int b = min(len, nnegibor);
  for (int i = 0; i < b; ++ i)
    for (int j = start; j < start + len - i - 1; ++ j)
	  if(getElement(dist_knn, bid, j, ntrain) < getElement(dist_knn, bid, j + 1, ntrain)){
	    double tmp_dist = getElement(dist_knn, bid, j, ntrain);
		setElement(dist_knn, bid, j, ntrain, getElement(dist_knn, bid, j + 1, ntrain));
		setElement(dist_knn, bid, j + 1, ntrain, tmp_dist);
		
		int tmp_ino = getElementInt(ino_knn, bid, j, ntrain);
		setElementInt(ino_knn, bid, j, ntrain, getElementInt(ino_knn, bid, j + 1, ntrain));
		setElementInt(ino_knn, bid, j + 1, ntrain, tmp_ino);
	  }

  __syncthreads();  

  __shared__ double dist[BSIZE];
  __shared__ int ino[BSIZE];
  __shared__ int shortest[BSIZE];
  
  int p = start + len -1;
  for (int i = 0; i < nnegibor; ++ i){
    if (b > 0){
      dist[tid] = getElement(dist_knn, bid, p, ntrain);
      ino[tid] = getElementInt(ino_knn, bid, p, ntrain);
	}
	else
      dist[tid] = DBL_MAX;
	
    shortest[tid] = tid;
  
	int stride = blockDim.x/2;
	while (stride > 0){
	  __syncthreads();
	  if (tid < stride){
		if (dist[tid] > dist[tid + stride]){
		  dist[tid] = dist[tid + stride];
		  ino[tid] = ino[tid + stride];
		  shortest[tid] = shortest[tid + stride];
		}
	  }
	  stride /= 2;
	}
	
	__syncthreads();
	if(tid == 0)
	  setElementInt(neighbor_knn, bid, i, nnegibor, ino[0]);
	if(tid == shortest[0]){
	  -- b;
	  -- p;
	}
  }
}

__global__ void knnMatching(){
  int ub = ntest * nnegibor;
  int stride = blockDim.x * gridDim.x;  
  int idx_test, idx_train;
  for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < ub; m += stride){
    idx_test = m / nnegibor;
	idx_train = neighbor_knn[m];
	if (label_test[idx_test] == label_train[idx_train])
	  neighbor_knn[m] = 1;
	else
	  neighbor_knn[m] = 0;
  }
}

// lauch with single block
__global__ void knnAcc(int neiborhood_size){
  int tid = threadIdx.x;
  int stride = blockDim.x;
  
  if (tid < 4)
    hits[tid] = 0;
	
  __shared__ int matched[BSIZE];
  matched[tid] = 0;
  
  for (int m = tid; m < ntest; m += stride){
    int nsametype = 0;
    for (int i = 0; i < neiborhood_size; ++ i)
	  nsametype += neighbor_knn[m * nnegibor + i];
	if (nsametype > neiborhood_size/2){
	  matched[tid] += 1;
	  if (label_test[m] == FN || label_test[m] == FP)
	    atomicAdd(&hits[label_test[m]], 1);
	}
	else{
	  if (label_test[m] == TN || label_test[m] == TP)
	    atomicSub(&hits[label_test[m]], 1);
	}
  }
  
  int stride1 = blockDim.x/2;
  while (stride1 > 0){
	__syncthreads();
	if (tid < stride1)
	  matched[tid] += matched[tid + stride1];
	stride1 /= 2;
  }
  
  __syncthreads();  
  if (tid ==0)
    acc_knn = 1.0 * matched[0] / ntest;
}

__global__ void knnUpdateDist_fortargetupdate(){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size = gridDim.x;
	
  for(int m = bid; m < ntrain * ntrain; m += size){
    int i = m / ntrain;
	int j = m % ntrain;
	double d = DBL_MAX;
	if (i != j && label_train[i] == label_train[j])
	  d = calcDist(i, km_train, j, km_train);
	if (tid == 0){
	  ino_knn[m] = j;
      dist_knn[m] = d;
	}
  }
}

__global__ void knnUpdateDist_train(){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size = gridDim.x;
	
  for(int m = bid; m < ntrain * ntrain; m += size){
    int i = m / ntrain;
	int j = m % ntrain;
	double d = DBL_MAX;
	if (i != j)
	  if (nclass == 2)
	    d = calcDist(i, km_train, j, km_train);
	  else
	    if (label_train[i] == label_train[j] || label_train[i] + label_train[j] == 3)
	      d = calcDist(i, km_train, j, km_train);
	if (tid == 0){
	  ino_knn[m] = j;
      dist_knn[m] = d;
	}
  }
}

// lauched with # block = ntrain
__global__ void knnFindNeighbor_train(){
  
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int len = ntrain / BSIZE;
  int start = tid * len;
  if (tid < ntrain % BSIZE){
    start += tid;
	++ len;
  }
  else
    start += ntrain % BSIZE;
  
  __syncthreads();
  int b = min(len, nnegibor);
  /* each thread sort its own chunk (start, len) by bubble sorting for b iterations.
     First b elements of ino_knn hold the closest b neighbors.*/
  for (int i = 0; i < b; ++ i)
    for (int j = start; j < start + len - i - 1; ++ j)
	  if(getElement(dist_knn, bid, j, ntrain) < getElement(dist_knn, bid, j + 1, ntrain)){
	    double tmp_dist = getElement(dist_knn, bid, j, ntrain);
		setElement(dist_knn, bid, j, ntrain, getElement(dist_knn, bid, j + 1, ntrain));
		setElement(dist_knn, bid, j + 1, ntrain, tmp_dist);
		
		int tmp_ino = getElementInt(ino_knn, bid, j, ntrain);
		setElementInt(ino_knn, bid, j, ntrain, getElementInt(ino_knn, bid, j + 1, ntrain));
		setElementInt(ino_knn, bid, j + 1, ntrain, tmp_ino);
	  }

  __syncthreads();  

  __shared__ double dist[BSIZE];
  __shared__ int ino[BSIZE];
  __shared__ int shortest[BSIZE];
  
  /* perform a merge sort of BSIZE sorted chunk. */
  int p = start + len -1;
  for (int i = 0; i < nnegibor; ++ i){
    if (b > 0){
      dist[tid] = getElement(dist_knn, bid, p, ntrain);
      ino[tid] = getElementInt(ino_knn, bid, p, ntrain);
	}
	else
      dist[tid] = DBL_MAX;
	
    shortest[tid] = tid;
  
	int stride = blockDim.x/2;
	while (stride > 0){
	  __syncthreads();
	  if (tid < stride){
		if (dist[tid] > dist[tid + stride]){
		  dist[tid] = dist[tid + stride];
		  ino[tid] = ino[tid + stride];
		  shortest[tid] = shortest[tid + stride];
		}
	  }
	  stride /= 2;
	}
	
	__syncthreads();
	if(tid == 0)
	  setElementInt(neighbor_knn, bid, i, nnegibor, ino[0]);
	if(tid == shortest[0]){
	  -- b;
	  -- p;
	}
  }
}


__global__ void knnMatching_train(){
  int ub = ntrain * nnegibor;
  int stride = blockDim.x * gridDim.x;  
  int idx_train1, idx_train2;
  for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < ub; m += stride){
    idx_train1 = m / nnegibor;
	idx_train2 = neighbor_knn[m];
	if (label_train[idx_train1] == label_train[idx_train2])
	  neighbor_knn[m] = 1;
	else
	  neighbor_knn[m] = 0;
  }
}

// lauch with single block
__global__ void knnAcc_train(int neiborhood_size){
  int tid = threadIdx.x;
  int stride = blockDim.x;
  
  if (tid < 4)
    hits[tid] = 0;
	
  __shared__ int matched[BSIZE];
  matched[tid] = 0;
  
  for (int m = tid; m < ntrain; m += stride){
    int nsametype = 0;
    for (int i = 0; i < neiborhood_size; ++ i)
	  nsametype += neighbor_knn[m * nnegibor + i];
	if (nsametype > neiborhood_size/2){
	  matched[tid] += 1;
	  if (label_train[m] == FN || label_train[m] == FP)
	    atomicAdd(&hits[label_train[m]], 1);
	}
	else{
	  if (label_train[m] == TN || label_train[m] == TP)
	    atomicSub(&hits[label_train[m]], 1);
	}
  }
  
  int stride1 = blockDim.x/2;
  while (stride1 > 0){
	__syncthreads();
	if (tid < stride1)
	  matched[tid] += matched[tid + stride1];
	stride1 /= 2;
  }
  
  __syncthreads();  
  if (tid ==0)
    acc_knn = 1.0 * matched[0] / ntrain;
}

///////////////////////////////////////////////////////////////////////

__global__ void knnUpdateDist1(Type eType){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size = gridDim.x;
  int ninst;
  double *km;
  short *label;
  switch (eType){
    case TRAIN:
	  ninst = ntrain;
	  km = km_train;
	  label = label_train;
	  break;
    case VALIDATE:
	  ninst = nvalidate;
	  km = km_validate;
	  label = label_validate;
	  break;
    case TEST:
	  ninst = ntest;
	  km = km_test;
	  label = label_test;
	  break;
  }
	
  for(int m = bid; m < ninst * ntrain; m += size){
    int i = m / ntrain;
	int j = m % ntrain;
	double d = DBL_MAX;
	if (nclass == 2){
	  if (eType != TRAIN || i != j)
	    d = calcDist(i, km, j, km_train);
	}
	else{
	  if (eType != TRAIN || i != j)
	    if (label[i] == label_train[j] || label_train[i] + label_train[j] == 3)
	      d = calcDist(i, km, j, km_train);
	}
	if (tid == 0){
	  ino_knn[m] = j;
      dist_knn[m] = d;
	}
  }
}

// lauched with # block = ntrain
__global__ void knnFindNeighbor1(){
  
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int len = ntrain / BSIZE;
  int start = tid * len;
  if (tid < ntrain % BSIZE){
    start += tid;
	++ len;
  }
  else
    start += ntrain % BSIZE;
  
  __syncthreads();
  int b = min(len, nnegibor);
  /* each thread sort its own chunk (start, len) by bubble sorting for b iterations.
     First b elements of ino_knn hold the closest b neighbors.*/
  for (int i = 0; i < b; ++ i)
    for (int j = start; j < start + len - i - 1; ++ j)
	  if(getElement(dist_knn, bid, j, ntrain) < getElement(dist_knn, bid, j + 1, ntrain)){
	    double tmp_dist = getElement(dist_knn, bid, j, ntrain);
		setElement(dist_knn, bid, j, ntrain, getElement(dist_knn, bid, j + 1, ntrain));
		setElement(dist_knn, bid, j + 1, ntrain, tmp_dist);
		
		int tmp_ino = getElementInt(ino_knn, bid, j, ntrain);
		setElementInt(ino_knn, bid, j, ntrain, getElementInt(ino_knn, bid, j + 1, ntrain));
		setElementInt(ino_knn, bid, j + 1, ntrain, tmp_ino);
	  }

  __syncthreads();  

  __shared__ double dist[BSIZE];
  __shared__ int ino[BSIZE];
  __shared__ int shortest[BSIZE];
  
  /* perform a merge sort of BSIZE sorted chunk. */
  int p = start + len -1;
  for (int i = 0; i < nnegibor; ++ i){
    if (b > 0){
      dist[tid] = getElement(dist_knn, bid, p, ntrain);
      ino[tid] = getElementInt(ino_knn, bid, p, ntrain);
	}
	else
      dist[tid] = DBL_MAX;
	
    shortest[tid] = tid;
  
	int stride = blockDim.x/2;
	while (stride > 0){
	  __syncthreads();
	  if (tid < stride){
		if (dist[tid] > dist[tid + stride]){
		  dist[tid] = dist[tid + stride];
		  ino[tid] = ino[tid + stride];
		  shortest[tid] = shortest[tid + stride];
		}
	  }
	  stride /= 2;
	}
	
	__syncthreads();
	if(tid == 0)
	  setElementInt(neighbor_knn, bid, i, nnegibor, ino[0]);
	if(tid == shortest[0]){
	  -- b;
	  -- p;
	}
  }
}


__global__ void knnMatching1(Type eType){
  int ninst;
  short *label;
  switch (eType){
    case TRAIN:
	  ninst = ntrain;
	  label = label_train;
	  break;
    case VALIDATE:
	  ninst = nvalidate;
	  label = label_validate;
	  break;
    case TEST:
	  ninst = ntest;
	  label = label_test;
	  break;
  }
	
  int ub = ninst * nnegibor;
  int stride = blockDim.x * gridDim.x;  
  int idx1, idx2;
  for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < ub; m += stride){
    idx1 = m / nnegibor;
	idx2 = neighbor_knn[m];
	if (label[idx1] == label_train[idx2])
	  neighbor_knn[m] = 1;
	else
	  neighbor_knn[m] = 0;
  }
}

// lauch with single block
__global__ void knnAcc1(int neiborhood_size, Type eType){
  int ninst;
  short *label;
  switch (eType){
    case TRAIN:
	  ninst = ntrain;
	  label = label_train;
	  break;
    case VALIDATE:
	  ninst = nvalidate;
	  label = label_validate;
	  break;
    case TEST:
	  ninst = ntest;
	  label = label_test;
	  break;
  }
  
  int tid = threadIdx.x;
  int stride = blockDim.x;
  
  if (tid < 4)
    hits[tid] = 0;
	
  __shared__ int matched[BSIZE];
  matched[tid] = 0;
  
  for (int m = tid; m < ninst; m += stride){
    int nsametype = 0;
    for (int i = 0; i < neiborhood_size; ++ i)
	  nsametype += neighbor_knn[m * nnegibor + i];
	if (nsametype > neiborhood_size/2){
	  matched[tid] += 1;
	  if (label[m] == FN || label[m] == FP)
	    atomicAdd(&hits[label[m]], 1);
	}
	else{
	  if (label[m] == TN || label[m] == TP)
	    atomicSub(&hits[label[m]], 1);
	}
  }
  
  int stride1 = blockDim.x/2;
  while (stride1 > 0){
	__syncthreads();
	if (tid < stride1)
	  matched[tid] += matched[tid + stride1];
	stride1 /= 2;
  }
  
  __syncthreads();  
  if (tid ==0)
    acc_knn = 1.0 * matched[0] / ninst;
}

///////////////////////////////////////////////////////////////////////

__global__ void updateTarget(){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size = gridDim.x * blockDim.x;
  int max_nn = max(max(nn[0], nn[1]), max(nn[2], nn[3]));
  for (int m = blockDim.x * bid + tid; m < ntrain * max_nn; m += size){
    int ino = m / max_nn;
	int idx_neighbor = m % max_nn;
    if (idx_neighbor < nn[label_train[ino]])
	  setTargetByOffset(ino, idx_neighbor, getElementInt(neighbor_knn, ino, idx_neighbor, nnegibor));
  }
}

__global__ void zeroTargetTerm(){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size = gridDim.x * blockDim.x;
  for (int m = blockDim.x * bid + tid; m < ntrain * ntrain; m += size)
    t_target[m] = .0;
}

__device__ void updateTar(int idx1, int idx2, double h){
  __syncthreads();
  for (int p = threadIdx.x; p < ntrain; p += blockDim.x)
    t_target[p * ntrain + idx1] += h * (getElement(km_train, idx1, p, ntrain) - getElement(km_train, idx2, p, ntrain));
}

__global__ void updateTargetTerm(){  
  int i, j;
  double h;
  int bid = blockIdx.x;
  
  for (i = 0; i < ntrain; ++ i){
    for (int kk = 0; kk < nn[label_train[i]]; ++ kk){
	  j = getTargetByOffset(i, kk);
	  
		h = d_nu[label_train[i]];
	    if (i % gridDim.x == bid)
		  updateTar(i, j, h);
	    if (j % gridDim.x == bid)
		  updateTar(j, i, h);

	}
  }
}

__global__ void countTarget(){
  __shared__ int stay[BSIZE*4];
  
  int tid = threadIdx.x;

  for (int i = 0; i < 4; ++ i)
    stay[tid + BSIZE * i] = 0;
  
  for(int m = tid; m < ntrain; m += BSIZE){
    int l = label_train[m];
	for (int i = 0; i < nn[l]; ++ i){
	  int n = getElementInt(neighbor_knn, m, i, nnegibor);
	  for (int j = 0; j < nn[l]; ++ j){
	    int t = getTargetByOffset(m, j);
	    if ( n == t)
		  ++ stay[l * BSIZE + tid];
	  }
	}
  }
  
  for (int i = 0; i < 4; ++ i){
    int stride1 = blockDim.x/2;
    while (stride1 > 0){
	  __syncthreads();
	  if (tid < stride1)
	    stay[BSIZE * i + tid] += stay[BSIZE * i + tid + stride1];
	  stride1 /= 2;
    }
    __syncthreads();
    if (tid == 0)
	  hits[i] = stay[BSIZE * i];
  }
}

void deviceInitKernelMatrix(int nf, int trainninst, int testninst, int validateninst, double *traindata, double *testdata, double *validatedata){

  //cudaMemcpyToSymbol(ntrain, trainninst, sizeof(int), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(ntest, testninst, sizeof(int), 0, cudaMemcpyHostToDevice);
  //cudaMemcpyToSymbol(nfeat, nf, sizeof(int), 0, cudaMemcpyHostToDevice);
  
  
  double *d_train_data, *d_test_data, *d_validate_data;
  cudaMalloc((void **)&d_train_data, sizeof(double) * trainninst * nf);
  cudaMemcpy(d_train_data, traindata, sizeof(double) * trainninst * nf, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_test_data, sizeof(double) * testninst * nf);
  cudaMemcpy(d_test_data, testdata, sizeof(double) * testninst * nf, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_validate_data, sizeof(double) * validateninst * nf);
  cudaMemcpy(d_validate_data, validatedata, sizeof(double) * validateninst * nf, cudaMemcpyHostToDevice);
  
  double *d_kernel_matrix_train, *d_kernel_matrix_test, *d_kernel_matrix_validate;
  cudaMalloc((void **)&d_kernel_matrix_train, sizeof(double) * trainninst * trainninst);
  cudaMemcpyToSymbol(km_train, &d_kernel_matrix_train, sizeof(double*), 0, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_kernel_matrix_test, sizeof(double) * testninst * trainninst);
  cudaMemcpyToSymbol(km_test, &d_kernel_matrix_test, sizeof(double*), 0, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_kernel_matrix_validate, sizeof(double) * validateninst * trainninst);
  cudaMemcpyToSymbol(km_validate, &d_kernel_matrix_validate, sizeof(double*), 0, cudaMemcpyHostToDevice);
  
  // Run the event recording
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event) ;
  cudaEventCreate(&stop_event) ;
  cudaEventRecord(start_event, 0);
  
  calcKM<<<84, 256>>>(d_train_data, d_test_data, d_validate_data);
  cudaThreadSynchronize();
  
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  
  cudaFree(d_train_data);
  cudaFree(d_test_data);
  cudaFree(d_validate_data);
}

void mallocToSymbol(size_t data_size, size_t offset, size_t pointer_size, const char *deviceSymbol){
  void *d_data;
  cudaMalloc((void **)&d_data, data_size);
  cudaMemcpyToSymbol(deviceSymbol, &d_data, pointer_size, offset, cudaMemcpyHostToDevice);
}

void dataToSymbol(void *data, size_t data_size, size_t offset, size_t pointer_size, const char *deviceSymbol){
  if (pointer_size > 0){
    void *d_data;
    cudaMalloc((void **)&d_data, data_size);
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSymbol, &d_data, pointer_size, offset, cudaMemcpyHostToDevice);
  }
  else
    cudaMemcpyToSymbol(deviceSymbol, data, data_size, offset, cudaMemcpyHostToDevice);
}

void deviceInitInstList(struct Inst *inst, unsigned *count, unsigned ninst){
  
  struct Inst *gi[4];
  for (int i = 0; i < 4; ++ i){
    if (count[i] > 0)
      gi[i] = (struct Inst *)malloc(sizeof(struct Inst) * count[i]);
  }

  int p[4] = {0, 0, 0, 0};
  for(int i = 0; i < ninst; ++ i){
    int type = inst[i].label;
	gi[type][p[type]].ino = inst[i].ino;
	gi[type][p[type]].label = inst[i].label;
	++ p[type];
  }
  
  struct Inst *d_inst;
  cudaMalloc((void **)&d_inst, sizeof(struct Inst) * ninst);
  unsigned start = 0;
  for (int i = 0; i < 4; ++ i){
    if (count[i] > 0)
	  cudaMemcpy(d_inst + start, gi[i], sizeof(struct Inst) * count[i], cudaMemcpyHostToDevice);
    struct Inst *dd_inst = d_inst + start;
    cudaMemcpyToSymbol(type_inst, &dd_inst, sizeof(struct Inst *), i * sizeof(struct Inst *), cudaMemcpyHostToDevice);
    start += count[i];
  }
  
  for (int i = 0; i < 4; ++ i){
    if (count[i] > 0)
      free(gi[i]);
  }
}


int totalMissed;
double targetCoverage[4];
double minCoverage;
int super = 0;

unsigned iter = 0;
unsigned n_target_update = 0;
/*
double global_max_acc = .0;
unsigned global_max_iter = 0;
unsigned global_max_pos = 0;
double global_max_acc_train = .0;
unsigned global_max_iter_train = 0;
unsigned global_max_pos_train = 0;
*/

double global_max_acc1[3] = {.0};
double global_max_pos1[3] = {.0};
double global_max_iter1[3] = {.0};

int targetUpdateNeeded(){
  if (super){
    super = 0;
	return 1;
  }
  if (alpha < 1e-8 && totalMissed > 0)
  //if ((alpha < 1e-8 && totalMissed > 0) || minCoverage < 0.5)
    return 1;
  else
    return 0;
}

void findNeighbor(Type eType){
  int ninst;
  switch (eType){
    case TRAIN:
	  ninst = n_train;
	  break;
    case VALIDATE:
	  ninst = n_validate;
	  break;
    case TEST:
	  ninst = n_test;
	  break;
  }
  knnUpdateDist1<<<84, BSIZE>>>(eType);
  knnFindNeighbor1<<<ninst, BSIZE>>>();
}

void calcAcc(Type eType){
  double dd[20];
  int h_hits[4];

  knnMatching1<<<84, BSIZE>>>(eType);

  double max_acc = .0;
  int max_acc_k = -1;
  for (int i = 0; i < 20; ++ i){
    knnAcc1<<<1, BSIZE>>>(2 * i + 1, eType);
	cudaThreadSynchronize();
	cudaMemcpyFromSymbol(h_hits, hits, sizeof(int) * 4, 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&dd[i], acc_knn, sizeof(double), 0, cudaMemcpyDeviceToHost);

	if (dd[i] > max_acc){
	  max_acc = dd[i];
	  max_acc_k = 2 * i + 1;
    }
    cout << h_hits[0] + h_hits[1] + h_hits[2] + h_hits[3] << "(" << h_hits[0] << "," << h_hits[1] << "," << h_hits[2] << "," << h_hits[3] << "), ";
  }
	
  //if (max_acc >= global_max_acc && iter > 10){
  if (max_acc >= global_max_acc1[eType]){
      global_max_acc1[eType] = max_acc;
	  global_max_iter1[eType] = iter;
	  global_max_pos1[eType] = max_acc_k;
  }
    cout << endl << "max acc = " << max_acc << " at k = " << max_acc_k 
    << ". global max = " << global_max_acc1[eType] << " in iter " << global_max_iter1[eType] << " at k = " << global_max_pos1[eType] << endl;
}

void findTarget(){
  int h_hits[4];
    countTarget<<<1, BSIZE>>>();
    cudaThreadSynchronize();
    cudaMemcpyFromSymbol(h_hits, hits, sizeof(int) * 4, 0, cudaMemcpyDeviceToHost);
    cout << "Targets: " 
	<< 1.0 * h_hits[0]/(tcount[0]*k[0]) << "(" << h_hits[0] << "/" << tcount[0]*k[0] << "), " 
	<< 1.0 * h_hits[1]/(tcount[1]*k[1]) << "(" << h_hits[1] << "/" << tcount[1]*k[1] << "), " 
	<< 1.0 * h_hits[2]/(tcount[2]*k[2]) << "(" << h_hits[2] << "/" << tcount[2]*k[2] << "), " 
	<< 1.0 * h_hits[3]/(tcount[3]*k[3]) << "(" << h_hits[3] << "/" << tcount[3]*k[3] << ")"<< endl ;	
	
	minCoverage = 1.0;
    for (int i = 0; i < 4; ++ i){
      targetCoverage[i] = 1.0 * h_hits[i] / (tcount[i]*k[i]);
	  if (minCoverage > targetCoverage[i])
	    minCoverage = targetCoverage[i];
    }

	totalMissed = 0;
    for (int i = 0; i < 4; ++ i)
      totalMissed += tcount[i] * k[i] - h_hits[i];
}
	  
void kernelTest(){
  char path[1024];
  getcwd(path, 1024);
  double original_alpha = alpha;
  double f, f_old = DBL_MAX;
  //int h_hits[4];
  //deviceInitKnn(n, n_test, 40);
    knnFindNeighbor_train<<<n_train, BSIZE>>>();
    updateTarget<<<84, BSIZE>>>();
	zeroTargetTerm<<<84, BSIZE>>>();
	updateTargetTerm<<<84, BSIZE>>>();
	
  unsigned min_iter = 0;  
  bool targetUpdated = false;
  int idx = 1;
  
  while(true){
  // Run the event recording
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);
  
  cout << endl << "Iter = " << iter << ", mu = " << mu << ", k = " << k[0] << "," << k[1] << "," << k[2] << "," << k[3] << ", nu = "  << nu[0] << "," << nu[1] << "," << nu[2] << "," << nu[3] << endl;  

  idx = 1 - idx;
  cudaMemcpyToSymbol(idx_o, &idx, sizeof(int), 0, cudaMemcpyHostToDevice);
  
  
  // update target and target term periodically
  //if (targetUpdateNeeded()){
  if (false){
    knnUpdateDist_fortargetupdate<<<84, BSIZE>>>();
    knnFindNeighbor_train<<<n_train, BSIZE>>>();
    updateTarget<<<84, BSIZE>>>();
	zeroTargetTerm<<<84, BSIZE>>>();
	updateTargetTerm<<<84, BSIZE>>>();
	alpha = original_alpha;
    targetUpdated = true;
	n_target_update ++;
  }
  
  // update distances to targets(i,j) and between opposing points(i,l)
  update2<<<84, 256>>>();
  
  // update t_triplet by calculating vdist of every (i, j, l)
  zeroT_triplet<<<84, 256>>>(); 
  update3_2<<<84, 256>>>();
  
  // update object function value
  calcFval<<<84, 256>>>();
  
  cudaThreadSynchronize();
  cudaMemcpyFromSymbol(&f, f_val, sizeof(double), 0, cudaMemcpyDeviceToHost);
  cout << "f_val= " << f;
  
  if (f < f_old || targetUpdated){
    targetUpdated = false;
	cout << ", reduced by " << f_old - f << endl;
    f_old = f;
	min_iter = iter;
    alpha *= 1.1;
	
	// test dataset
	findNeighbor(TEST);
	calcAcc(TEST);
	
	// validate dataset
	findNeighbor(VALIDATE);
	calcAcc(VALIDATE);
	
	// train dataset
	findNeighbor(TRAIN);
	findTarget();
	/*
    countTarget<<<1, BSIZE>>>();
    cudaThreadSynchronize();
    cudaMemcpyFromSymbol(h_hits, hits, sizeof(int) * 4, 0, cudaMemcpyDeviceToHost);
    cout << "Targets: " 
	<< 1.0 * h_hits[0]/(tcount[0]*k[0]) << "(" << h_hits[0] << "/" << tcount[0]*k[0] << "), " 
	<< 1.0 * h_hits[1]/(tcount[1]*k[1]) << "(" << h_hits[1] << "/" << tcount[1]*k[1] << "), " 
	<< 1.0 * h_hits[2]/(tcount[2]*k[2]) << "(" << h_hits[2] << "/" << tcount[2]*k[2] << "), " 
	<< 1.0 * h_hits[3]/(tcount[3]*k[3]) << "(" << h_hits[3] << "/" << tcount[3]*k[3] << ")"<< endl ;	
	
	minCoverage = 1.0;
    for (int i = 0; i < 4; ++ i){
      targetCoverage[i] = 1.0 * h_hits[i] / (tcount[i]*k[i]);
	  if (minCoverage > targetCoverage[i])
	    minCoverage = targetCoverage[i];
    }

	totalMissed = 0;
    for (int i = 0; i < 4; ++ i)
      totalMissed += tcount[i] * k[i] - h_hits[i];
    */
	calcAcc(TRAIN);	
  }
  else{
	cout << ", increased by " << f - f_old;
    alpha /= 10;
    copyO<<<84, BSIZE>>>();
    update2<<<84, 256>>>();
    zeroT_triplet<<<84, 256>>>();
    update3_2<<<84, 256>>>();
  }
  
  cout << "min_f = " << f_old << " at iter " << min_iter << ", alpha = " << alpha << " target udpated = " << n_target_update << endl;

  // t_update = I - 2 * alpha * (t_target + t_triplet)
  updateUpdateTerm<<<84, 256>>>(alpha);  
  
  // update omega = omega * t_update
  zeroO<<<84, 256>>>();
  dim3 dimGrid(n_feat, (n_train - 1) / BSIZE + 1);
  dim3 dimBlock(BSIZE);
  updateO1<<<dimGrid, dimBlock>>>();  
  cudaThreadSynchronize();
  
  float time_kernel;
  cudaEventRecord(stop_event, 0);
  cudaEventElapsedTime(&time_kernel, start_event, stop_event);
  cout << "time " << time_kernel/1000 << " at " << path << endl;
  ++ iter;
  //if (iter > 100)
  if (alpha < 1e-10)
    break;
  }
}
