#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

#include "segmerge.h"

#define CUDA_CHECK(_e, _s) if(_e != cudaSuccess) { \
  std::cout << "CUDA error (" << _s << "): " \
            << cudaGetErrorString(_e) \
            << std::endl; \
  return 0; }

int main (int argc, char* argv[]) {

  using K = int;
  using T = int;

  /* if (argc != 3) {
    std::cerr << "usage: ./run n_a n_b\n";
    std::exit(1);
  }

  // params
  int n_a = std::atoi(argv[1]); // size of array A
  int n_b = std::atoi(argv[2]); // size of array B */


  if (argc != 2) {
    std::cerr << "usage: ./run n_a\n";
    std::exit(1);
  }

  // params
  int n_a = std::atoi(argv[1]); // size of array A
  int n_b = n_a * 2; // size of array B

  int largest_key = 20;
  int max_seg_size = 10;
  int min_seg_size = 0;
  
  // info
  std::cout << "Merging two arrays of " << n_a << " and " << n_b << " keys and vals\n";
  std::cout << "Largest key       : " << largest_key << std::endl;
  std::cout << "Smallest key      : " << 0 << std::endl;
  std::cout << "Largest value     : " << 1 << std::endl;
  std::cout << "Smallest value    : " << 1 << std::endl;
  std::cout << "Largest seg size  : " << max_seg_size << std::endl;
  std::cout << "Smallest seg size : " << min_seg_size << std::endl; 

  // seed
#ifdef DEBUG
  srand(123);
#else
  srand(static_cast<unsigned>(time(0)));
#endif

  // create arrays for key_a, key_b, val_a, val_b
  std::vector<K> key_a(n_a);
  std::vector<K> key_b(n_b);
  std::vector<K> key_c;
  std::vector<T> val_a(n_a);
  std::vector<T> val_b(n_b);
  std::vector<T> val_c;

  // populate arrays
  for (std::size_t i = 0; i < n_a; i++) {
    key_a[i] = std::rand() % largest_key;
    val_a[i] = 1; // can be randomized as well
  }
  for (std::size_t i = 0; i < n_b; i++) {
    key_b[i] = std::rand() % largest_key;
    val_b[i] = 1; // can be randomized as well
  }

  // Segmentations
  std::vector<int> seg_a, seg_b, seg_c;
  int start_a = 0, start_b = 0;
  
  while (start_a < n_a) {
    seg_a.emplace_back(start_a);
    int sz = std::rand() % max_seg_size + min_seg_size;
    start_a = seg_a.back() + sz;
  }

  while (start_b < n_b) {
    seg_b.emplace_back(start_b);
    int sz = std::rand() % max_seg_size + min_seg_size;
    start_b = seg_b.back() + sz;
  }

  int m_a = seg_a.size();
  int m_b = seg_b.size();
  int m_c = max(m_a, m_b);
  seg_c.emplace_back(0);
  
  // allocate GPU memory
  cudaError_t err;
  K* key_a_d;
  K* key_b_d;
  K* key_c_d;
  T* val_a_d;
  T* val_b_d;
  T* val_c_d;
  int* seg_a_d;
  int* seg_b_d;
  int* seg_c_d;

  err = cudaMalloc(&key_a_d, sizeof(K)*n_a);
  CUDA_CHECK(err, "alloc key_a_d");
  err = cudaMalloc(&key_b_d, sizeof(K)*n_b);
  CUDA_CHECK(err, "alloc key_b_d");
  err = cudaMalloc(&key_c_d, sizeof(K)*(n_a + n_b)); // At most size of A + B
  CUDA_CHECK(err, "alloc key_c_d");
  err = cudaMalloc(&val_a_d, sizeof(T)*n_a);
  CUDA_CHECK(err, "alloc val_a_d");
  err = cudaMalloc(&val_b_d, sizeof(T)*n_b);
  CUDA_CHECK(err, "alloc val_b_d");
  err = cudaMalloc(&val_c_d, sizeof(T)*(n_a + n_b));
  CUDA_CHECK(err, "alloc val_c_d");
  err = cudaMalloc(&seg_a_d, sizeof(int)*m_a);
  CUDA_CHECK(err, "alloc seg_a_d");
  err = cudaMalloc(&seg_b_d, sizeof(int)*m_b);
  CUDA_CHECK(err, "alloc seg_b_d");
  err = cudaMalloc(&seg_c_d, sizeof(int)*m_c);
  CUDA_CHECK(err, "alloc seg_c_d");

  // copy data from host to device
  err = cudaMemcpy(key_a_d, &key_a[0], sizeof(K)*n_a, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "copy to key_a_d"); 
  err = cudaMemcpy(key_b_d, &key_b[0], sizeof(K)*n_b, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "copy to key_b_d"); 
  err = cudaMemcpy(val_a_d, &val_a[0], sizeof(T)*n_a, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "copy to val_a_d"); 
  err = cudaMemcpy(val_b_d, &val_b[0], sizeof(T)*n_b, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "copy to val_b_d"); 
  err = cudaMemcpy(seg_a_d, &seg_a[0], sizeof(int)*m_a, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "copy to seg_a_d"); 
  err = cudaMemcpy(seg_b_d, &seg_b[0], sizeof(int)*m_b, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "copy to seg_b_d");

  auto begin = std::chrono::steady_clock::now();
  int new_n = segmerge(
    key_a_d, key_b_d, key_c_d,
    val_a_d, val_b_d, val_c_d,
    seg_a_d, seg_b_d, seg_c_d, 
    n_a, n_b, m_a, m_b
  );
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  std::cout << "CUDA runtime (us) : " <<
    std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
    << std::endl;

  // copy data from device to host
  std::vector<int> seg_c_h(m_c);
  std::vector<K> key_c_h(new_n); 
  std::vector<T> val_c_h(new_n); 
  err = cudaMemcpy(&seg_c_h[0], seg_c_d, sizeof(int)*m_c, cudaMemcpyDeviceToHost);
  CUDA_CHECK(err, "copy from seg_c_d");
  err = cudaMemcpy(&key_c_h[0], key_c_d, sizeof(K)*new_n, cudaMemcpyDeviceToHost); 
  CUDA_CHECK(err, "copy from key_c_d");
  err = cudaMemcpy(&val_c_h[0], val_c_d, sizeof(T)*new_n, cudaMemcpyDeviceToHost);
  CUDA_CHECK(err, "copy from val_c_d");

  print(seg_a, key_a);
  print(seg_b, key_b);
  print(seg_c_h, key_c_h);
  // int curseg = 1;
  // for (int i = 0; i < n_a + n_b ; i++ ) {
  //   if (i == seg_c_h[curseg]) {
  //     curseg++;
  //     printf(" SEG ");
  //   }
  //   printf("%d ",key_c_h[i]);
  // }
  // printf("\n");

  // free GPU memory
  err = cudaFree(key_a_d);
  CUDA_CHECK(err, "free key_a_d");
  err = cudaFree(key_b_d);
  CUDA_CHECK(err, "free key_b_d");
  err = cudaFree(key_c_d);
  CUDA_CHECK(err, "free key_c_d");
  err = cudaFree(val_a_d);
  CUDA_CHECK(err, "free val_a_d");
  err = cudaFree(val_b_d);
  CUDA_CHECK(err, "free val_b_d"); 
  err = cudaFree(val_c_d);
  CUDA_CHECK(err, "free val_c_d");
  err = cudaFree(seg_a_d);
  CUDA_CHECK(err, "free seg_a_d");
  err = cudaFree(seg_b_d);
  CUDA_CHECK(err, "free seg_b_d");
  err = cudaFree(seg_c_d);
  CUDA_CHECK(err, "free seg_c_d");

  // CPU-based validation
  begin = std::chrono::steady_clock::now();
  gold_segsort(key_a, val_a, n_a, seg_a, m_a);
  gold_segsort(key_b, val_b, n_b, seg_b, m_b);
  gold_segmerge(key_a, key_b, key_c,
                val_a, val_b, val_c,
                n_a, n_b, seg_a, seg_b, seg_c);
  print(seg_c, key_c);
  end = std::chrono::steady_clock::now();
  std::cout << "CPU runtime (us) : " <<
    std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
    << std::endl;

  // check
  int cnt = 0;

  // Compare segmentations
  for (std::size_t i = 0; i < seg_c.size(); i++) {
    if (seg_c[i] != seg_c_h[i]) cnt++;
  }
  if (cnt != 0) {
    std::cout << "[NOT PASSED] checking segs: #err = " << cnt << std::endl;
  } else {
    std::cout << "[PASSED] checking segs\n";
  }

  // Reset count for keys comparison
  cnt = 0;
  for (std::size_t i = 0; i < new_n; i++) {
    if (key_c[i] != key_c_h[i]) cnt++;
  }
  if (cnt != 0) {
    std::cout << "[NOT PASSED] checking keys: #err = " << cnt << std::endl;
  } else {
    std::cout << "[PASSED] checking keys\n";
  }

  // Reset count for values comparison
  cnt = 0;
  for (std::size_t i = 0; i < new_n; i++) {
    if (val_c[i] != val_c_h[i]) cnt++;
  }
  if (cnt != 0) {
    std::cout << "[NOT PASSED] checking vals: #err = " << cnt << std::endl;
  } else {
    std::cout << "[PASSED] checking vals\n";
  }

  // print
  //std::cout << "key_c:\n"; 
  //print(seg_c, key_c);
  //std::cout << "val_c:\n"; 
  //print(seg_c, val_c);
  //std::cout << "key_c_h:\n";
  //print(seg_c_h, key_c_h);
  //std::cout << "val_c_h:\n"; 
  //print(seg_c_h, val_c_h);

  return 0;
}


