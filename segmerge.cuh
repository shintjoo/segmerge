#include <cuda.h>

/* // Original filln
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

/* //updated filln
template<typename K, typename T>
__global__ void filln(
  K* key_a, K* key_b, K* key_c,
  T* val_a, T* val_b, T* val_c,
  int* seg_a, int* seg_b, int* seg_c,
  int n_a, int n_b, int m_a, int m_b)
{
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < m_a && tid < m_b) {
    int beg_a = seg_a[tid];
    int end_a = (tid + 1) < m_a ? seg_a[tid+1] : n_a;

    int beg_b = seg_b[tid];
    int end_b = (tid + 1) < m_b ? seg_b[tid+1] : n_b;

    std::size_t i = beg_a;
    std::size_t j = beg_b;
    std::size_t k = beg_a + beg_b;

    while (i < end_a || j < end_b) {
      if (i < end_a) {
        key_c[k] = key_a[i];
        val_c[k] = val_a[i];
        i++;
        k++;
      }
      if (j < end_b) {
        key_c[k] = key_b[j];
        val_c[k] = val_b[j];
        j++;
        k++;
      }
    }
    seg_c[tid] = beg_a + beg_b;
  } else if (tid < max(m_a, m_b)) { // Add on the final segment(s?)
    int beg;
    int end;
    int *key;
    int *val;
    int n;
    if (m_a > m_b) {
      beg = seg_a[tid];
      end = (tid + 1) < m_a ? seg_a[tid+1] : n_a;
      key = key_a;
      val = val_a;
      n = n_b;
    } else {
      beg = seg_b[tid];
      end = (tid + 1) < m_b ? seg_b[tid+1] : n_b;
      key = key_b;
      val = val_b;
      n = n_a;
    }
    int k = n + beg;
    seg_c[tid] = k;
    int i = beg;
    while (i < end) {
      key_c[k] = key[i];
      val_c[k] = val[i];
      i++;
      k++;
    }
    
  }
}

// Original merge
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

*/

// Attempt to merge filln and merge
template<typename K, typename T>
__global__ void filln_merge(
  K* key_a, K* key_b, K* key_c,
  T* val_a, T* val_b, T* val_c,
  int* seg_a, int* seg_b, int* seg_c,
  int* count,
  int n_a, int n_b, int m_a, int m_b)
{
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < max(m_a, m_b)) {
    // Determine segment boundaries for both arrays
    int beg_a = (tid < m_a) ? seg_a[tid] : n_a;
    int end_a = (tid + 1 < m_a) ? seg_a[tid + 1] : n_a;
    int beg_b = (tid < m_b) ? seg_b[tid] : n_b;
    int end_b = (tid + 1 < m_b) ? seg_b[tid + 1] : n_b;

    int i = beg_a;
    int j = beg_b;
    int k = beg_a + beg_b;
    int du = 0;

    K currentKey = 0;
    T currentSum = 0;
    bool hasCurrent = false;

    // Merge and fill simultaneously
    while (i < end_a || j < end_b) {
      K key_i = (i < end_a) ? key_a[i] : INT_MAX;
      K key_j = (j < end_b) ? key_b[j] : INT_MAX;

      K selectedKey;
      T selectedVal;

      if (key_i < key_j) {
        selectedKey = key_i;
        selectedVal = val_a[i++];
      } else if (key_i > key_j) {
        selectedKey = key_j;
        selectedVal = val_b[j++];
      } else {
        selectedKey = key_i;
        selectedVal = val_a[i++] + val_b[j++];
      }

      // Merge duplicates on the fly
      if (!hasCurrent || selectedKey != currentKey) {
        if (hasCurrent) {
          key_c[k] = currentKey;
          val_c[k] = currentSum;
          k++;
        }
        currentKey = selectedKey;
        currentSum = selectedVal;
        hasCurrent = true;
      } else {
        currentSum += selectedVal;
        du++;
      }
    }

    // Write the last merged key-value pair
    if (hasCurrent) {
      key_c[k] = currentKey;
      val_c[k] = currentSum;
      k++;
    }

    // Update segment and duplicate count
    seg_c[tid] = beg_a + beg_b;
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
