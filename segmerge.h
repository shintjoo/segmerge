#pragma once

#include <map>
#include <algorithm>
#include "bb_segsort/bb_segsort.h"
#include "segmerge.cuh"

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

// #define DEBUG
// #define SKIP_CPU
// #define SKIP_GPU

/* template<typename K, typename T>
void gold_segmerge(
  std::vector<K>& key_a, 
  std::vector<K>& key_b,
  std::vector<K>& key_c,
  std::vector<T>& val_a,
  std::vector<T>& val_b,
  std::vector<K>& val_c,
  int n, 
  const std::vector<int>& seg, 
  int m,
  std::vector<int>& seg_c
);

template<typename K, typename T>
void merge(
  std::vector<K>& key_a,
  std::vector<K>& key_b,
  std::vector<K>& key_c,
  std::vector<T>& val_a,
  std::vector<T>& val_b,
  std::vector<T>& val_c,
  int begin, int end,
  std::map<K, T>& mergedMap,
  std::vector<int>& seg_c
);

template<typename K, typename T>
void gold_segsort(
  std::vector<K>& key, 
  std::vector<T>& val, 
  int n, 
  const std::vector<int>& seg, 
  int m
);

template<typename K, typename T>
int segmerge(
  K* key_a_d, K* key_b_d, K* key_c_d,
  T* val_a_d, T* val_b_d, T* val_c_d,
  int* seg_d, int* seg_c_d, 
  int n, int m
);  */

template<typename K, typename T>
void gold_segmerge(
  std::vector<K>& key_a,
  std::vector<K>& key_b,
  std::vector<K>& key_c,
  std::vector<T>& val_a,
  std::vector<T>& val_b,
  std::vector<T>& val_c,
  int n_a,
  int n_b,
  std::vector<int>& seg_a,
  std::vector<int>& seg_b,
  std::vector<int>& seg_c
);

template<typename K, typename T>
void merge(
  std::vector<K>& key_a,
  std::vector<K>& key_b,
  std::vector<K>& key_c,
  std::vector<T>& val_a,
  std::vector<T>& val_b,
  std::vector<T>& val_c,
  int begin_a, int end_a,
  int begin_b, int end_b,
  std::map<K, T>& merged_map,
  std::vector<int>& seg_c
);

template<typename K, typename T>
void gold_segsort(
  std::vector<K>& key,
  std::vector<T>& val,
  int n,
  const std::vector<int>& seg, 
  int m
);

template<typename K, typename T>
int segmerge(
  K* key_a_d, K* key_b_d, K* key_c_d,
  T* val_a_d, T* val_b_d, T* val_c_d,
  int* seg_a_d, int* seg_b_d, int* seg_c_d,
  int n_a, int n_b, int m_a, int m_b
);

void print(
  const std::vector<int>& seg,
  const std::vector<int>& key);

///////////////////////////////////////
///////////////////////////////////////

/**************************************
 * for cpu segmented merge 
 * input:
 *   key_a: keys of array A
 *   key_b: keys of array B
 *   val_a: vals of array A 
 *   val_b: vals of array B
 *   n    : length of array A and B (int)
 *   seg  : segs of array A and B
 *   m    : number of segments (int)
 * output:
 *   key_c: keys of merged array C
 *   val_c: vals of merged array C
 *   seg_c: segs of merged array C  
 **************************************/
/* template<typename K, typename T>
void gold_segmerge(
  std::vector<K>& key_a, 
  std::vector<K>& key_b,
  std::vector<K>& key_c,
  std::vector<T>& val_a,
  std::vector<T>& val_b,
  std::vector<K>& val_c,
  int n, 
  const std::vector<int>& seg, 
  int m,
  std::vector<int>& seg_c)
{
  std::map<K, T> mergedMap;
  for (std::size_t i = 0; i < m; i++) {
    int end = (i<m-1) ? seg[i+1] : n;
    merge<K, T>(key_a, key_b, key_c,
          val_a, val_b, val_c,
          seg[i], end,
          mergedMap, seg_c);
  }
  seg_c.pop_back();
} */

template<typename K, typename T>
void gold_segmerge(
  std::vector<K>& key_a,
  std::vector<K>& key_b,
  std::vector<K>& key_c,
  std::vector<T>& val_a,
  std::vector<T>& val_b,
  std::vector<T>& val_c,
  int n_a,
  int n_b,
  std::vector<int>& seg_a,
  std::vector<int>& seg_b,
  std::vector<int>& seg_c
) {
  std::map<K, T> merged_map;
  int m_a = seg_a.size();
  int m_b = seg_b.size();

  for (int i = 0; i < std::max(m_a, m_b); i++) {
    int begin_a = (i < m_a) ? seg_a[i] : n_a;
    int end_a = (i + 1 < m_a) ? seg_a[i + 1] : n_a;
    int begin_b = (i < m_b) ? seg_b[i] : n_b;
    int end_b = (i + 1 < m_b) ? seg_b[i + 1] : n_b;

    merge<K, T>(
      key_a, key_b, key_c,
      val_a, val_b, val_c,
      begin_a, end_a,
      begin_b, end_b,
      merged_map, seg_c
    );
  }
  seg_c.pop_back();
}

/* template<typename K, typename T>
void merge(
  std::vector<K>& key_a,
  std::vector<K>& key_b,
  std::vector<K>& key_c,
  std::vector<T>& val_a,
  std::vector<T>& val_b,
  std::vector<T>& val_c,
  int begin, int end,
  std::map<K, T>& mergedMap,
  std::vector<int>& seg_c)
{
  mergedMap.clear();
  for (std::size_t i = begin; i < end; i++) {
    mergedMap[key_a[i]] += val_a[i];
    mergedMap[key_b[i]] += val_b[i];
  }

  for (const auto& [key, val] : mergedMap) {
    key_c.emplace_back(key);
    val_c.emplace_back(val);
  }
  int last = seg_c.back();
  seg_c.emplace_back(last+mergedMap.size());
} */

template<typename K, typename T>
void merge(
  std::vector<K>& key_a,
  std::vector<K>& key_b,
  std::vector<K>& key_c,
  std::vector<T>& val_a,
  std::vector<T>& val_b,
  std::vector<T>& val_c,
  int begin_a, int end_a,
  int begin_b, int end_b,
  std::map<K, T>& merged_map,
  std::vector<int>& seg_c
) {
  merged_map.clear();
  for (int i = begin_a; i < end_a; i++) {
    merged_map[key_a[i]] += val_a[i];
  }
  for (int i = begin_b; i < end_b; i++) {
    merged_map[key_b[i]] += val_b[i];
  }

  for (auto& [key, val] : merged_map) {
    key_c.push_back(key);
    val_c.push_back(val);
  }
  int last = seg_c.empty() ? 0 : seg_c.back();
  seg_c.push_back(last + merged_map.size());
}

template<typename K, typename T>
void gold_segsort(
  std::vector<K>& key, 
  std::vector<T>& val, 
  int n, 
  const std::vector<int>& seg, 
  int m)
{
  std::vector<std::pair<K,T>> pairs;
  for(int i = 0; i < n; i++) {
    pairs.push_back({key[i], val[i]});
  }

  for(int i = 0; i < m; i++) {
    int st = seg[i];
    int ed = (i<m-1) ? seg[i+1] : n;
      stable_sort(
        pairs.begin() + st, 
        pairs.begin() + ed, 
        [&](std::pair<K,T> a, std::pair<K,T> b){ return a.first < b.first;});
  }
    
  for(int i = 0; i < n; i++) {
    key[i] = pairs[i].first;
    val[i] = pairs[i].second;
  }
}

/* template<typename K, typename T>
int segmerge(
  K* key_a_d, K* key_b_d, K* key_c_d,
  T* val_a_d, T* val_b_d, T* val_c_d,
  int* seg_d, int* seg_c_d, int n, int m) 
{
  bb_segsort(key_a_d, val_a_d, n, seg_d, m);
  bb_segsort(key_b_d, val_b_d, n, seg_d, m);

  unsigned num_threads = 256;
  unsigned num_blocks = (m + 256 + 1) / 256;
  
  filln<<<num_blocks, num_threads>>>(
    key_a_d, key_b_d, key_c_d,
    val_a_d, val_b_d, val_c_d,
    seg_d, seg_c_d, n, m
  );

  bb_segsort(key_c_d, val_c_d, 2*n, seg_c_d, m);

  int* count;
  cudaMalloc(&count, sizeof(int)*m);
  merge<<<num_blocks, num_threads>>>(key_c_d, val_c_d, seg_c_d, count, 2*n, m);

  thrust::device_ptr<int> key_c_ptr(key_c_d);
  thrust::device_ptr<int> val_c_ptr(val_c_d);
  auto key_new_end = thrust::remove(thrust::device, key_c_ptr, key_c_ptr + 2*n, -1);
  auto val_new_end = thrust::remove(thrust::device, val_c_ptr, val_c_ptr + 2*n, -1);
  thrust::device_ptr<int> count_ptr(count);
  thrust::exclusive_scan(count_ptr, count_ptr+m, count_ptr);
  sub<<<num_blocks, num_threads>>>(seg_c_d, count, m);

  cudaFree(count);
  
  return key_new_end - key_c_ptr;
} */

template<typename K, typename T>
int segmerge(
  K* key_a_d, K* key_b_d, K* key_c_d,
  T* val_a_d, T* val_b_d, T* val_c_d,
  int* seg_a_d, int* seg_b_d, int* seg_c_d,
  int n_a, int n_b, int m_a, int m_b
) {

  unsigned num_threads = 256;
  unsigned num_blocks = (std::max(m_a, m_b) + num_threads - 1) / num_threads;

  filln<<<num_blocks, num_threads>>>(
    key_a_d, key_b_d, key_c_d,
    val_a_d, val_b_d, val_c_d,
    seg_a_d, seg_b_d, seg_c_d,
    n_a, n_b, m_a, m_b
  );

  bb_segsort(key_c_d, val_c_d, n_a + n_b, seg_c_d, std::max(m_a, m_b));

  int* count;
  cudaMalloc(&count, sizeof(int) * std::max(m_a, m_b));
  merge<<<num_blocks, num_threads>>>(
    key_c_d, val_c_d, seg_c_d, count, n_a + n_b, std::max(m_a, m_b)
  );

  thrust::device_ptr<K> key_c_ptr(key_c_d);
  thrust::device_ptr<T> val_c_ptr(val_c_d);
  auto key_new_end = thrust::remove(thrust::device, key_c_ptr, key_c_ptr + (n_a + n_b), -1);
  auto val_new_end = thrust::remove(thrust::device, val_c_ptr, val_c_ptr + (n_a + n_b), -1);

  thrust::device_ptr<int> count_ptr(count);
  thrust::exclusive_scan(count_ptr, count_ptr + std::max(m_a, m_b), count_ptr);
  sub<<<num_blocks, num_threads>>>(seg_c_d, count, std::max(m_a, m_b));

  cudaFree(count);

  return key_new_end - key_c_ptr;
}

void print(
  const std::vector<int>& seg,
  const std::vector<int>& key)
{
#ifdef DEBUG
  for (int i = 0; i < seg.size(); i++) {
    int end = (i+1)<seg.size() ? seg[i+1] : key.size();
    std::cout << "[ ";
    for (int j = seg[i]; j < end; j++) {
      std::cout << key[j] << " ";
    }
    std::cout << "] ";
  }
  std::cout << std::endl;
#endif
}
