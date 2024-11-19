#include <cuda.h>
/* 
template<typename K, typename T>
__global__ void filln(
  K* key_a, K* key_b, K* key_c,
  T* val_a, T* val_b, T* val_c,
  int* seg, int* seg_c, 
  int n, int m)
{
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < m) {
    int beg = seg[tid];
    int end = (tid + 1) < m ? seg[tid+1] : n;
    int sz = end - beg;
    std::size_t i = beg;
    std::size_t j = beg*2;
    std::size_t k = beg*2+sz;
    while (i < end && j < end*2 && k < end*2) {
      key_c[j] = key_a[i];
      val_c[j] = val_a[i];
      key_c[k] = key_b[i];
      val_c[k] = val_b[i];
      i++; j++; k++;
    }
    seg_c[tid] = beg * 2; 
  }
} */

template<typename K, typename T>
__global__ void filln(
  K* key_a, K* key_b, K* key_c,
  T* val_a, T* val_b, T* val_c,
  int* seg_a, int* seg_b, int* seg_c, 
  int n, int m_a, int m_b)
{
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < m_a || tid < m_b) {
    int beg_a = (tid < m_a) ? seg_a[tid] : n;
    int end_a = (tid + 1 < m_a) ? seg_a[tid + 1] : n;

    int beg_b = (tid < m_b) ? seg_b[tid] : n;
    int end_b = (tid + 1 < m_b) ? seg_b[tid + 1] : n;

    // Calculate the combined segment range
    int beg_c = min(beg_a, beg_b);
    int end_c = max(end_a, end_b);

    // Initialize pointers for merging
    int i = beg_a;
    int j = beg_b;
    int k = beg_c;

    // Merge values from A and B into C
    while (i < end_a && j < end_b) {
      if (key_a[i] < key_b[j]) {
        key_c[k] = key_a[i];
        val_c[k] = val_a[i];
        i++;
      } else {
        key_c[k] = key_b[j];
        val_c[k] = val_b[j];
        j++;
      }
      k++;
    }

    // Copy remaining elements from A
    while (i < end_a) {
      key_c[k] = key_a[i];
      val_c[k] = val_a[i];
      i++;
      k++;
    }

    // Copy remaining elements from B
    while (j < end_b) {
      key_c[k] = key_b[j];
      val_c[k] = val_b[j];
      j++;
      k++;
    }

    // Store new segment boundary
    seg_c[tid] = beg_c;
  }
}

template<typename K, typename T>
__global__ void merge(
  K* key, T* val, int* seg, int* count, int n, int m)
{
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < m) {
    int beg = seg[tid]; 
    int end = (tid + 1) < m ? seg[tid+1] : n;
    int du = 0;
    K currentKey = 0;
    T currentSum = 0;
    std::size_t i = beg;
    std::size_t j = i + 1;
    while (i < end && j < end) { 
      currentKey = key[i];
      currentSum = val[i];
      while (key[j] == currentKey && j < end) {
        currentSum += val[j];
        key[j] = -1;
        val[j] = -1;
        j++;
        du++;
      }
      val[i] = currentSum;
      i = j;
      j++;
    }
    count[tid] = du; 
  }
}

__global__ void sub(int* seg, int* count, int m)
{
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < m) {
    seg[tid] -= count[tid];
  }
}
