#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {
  ///构造函数，初始化Layer参数 数据变换器参数
template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) { //通过判断Blob个数选择是否输出lable
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();//生成随机种子
  // The subclasses should setup the size of bottom and top
  //子类负责设置Top Blob形状
  DataLayerSetUp(bottom, top);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_(param.data_param().prefetch()),
      prefetch_free_(), prefetch_full_(), prefetch_current_() {
  for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i].reset(new Batch<Dtype>());
    prefetch_free_.push(prefetch_[i].get());
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i]->data_.mutable_cpu_data();//这里调用的mutable_cpu_data是syncedmen中的函数，其中包括了内存的分配与设置
    if (this->output_labels_) {
      prefetch_[i]->label_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < prefetch_.size(); ++i) {
      prefetch_[i]->data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i]->label_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();//对数据与处理变换器进行随机初始化
  StartInternalThread();//开启内部预存取线程
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) { //循环载入
      Batch<Dtype>* batch = prefetch_free_.pop();//拿到一个空闲的batch并在空闲队列中删除该batch的元素
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        if (this->output_labels_) {
          batch->label_.data().get()->async_gpu_push(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);//把batch加入到以加载的队列中
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu( //拷贝数据
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);//根据Batch更改batch的形状
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());//获得batch的数据
  if (this->output_labels_) {//如果需要标签
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);//更改标签层的形状
    top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());//拷贝标签层的数据
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);//不知所云 ？？？？？
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
