#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  const void* cpu_data();//CPU上的data
  void set_cpu_data(void* data);//将cpu的data指针指向一个新的区域 并将原来申请的内存释放
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();//获得CPU的数据地址
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };//标识相应的数据状态
  SyncedHead head() const { return head_; }//返回相应的数据状态
  size_t size() const { return size_; }//返回数据大小

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);//一个cuda拷贝的异步传输，从cpu到gpu
#endif

 private:
  void check_device();

  void to_cpu();//数据从gpu拷贝到CPU 之后会改变head的标识
  void to_gpu();
  void* cpu_ptr_;//cpu数据的地址
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;//拥有CPU数据的标识
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;//拥有GPU数据的标识
  int device_;//gpu的ID号

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);//禁止该类的拷贝与赋值
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
