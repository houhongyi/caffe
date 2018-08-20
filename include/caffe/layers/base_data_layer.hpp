#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  //LayerSetup函数进行一般层的设置，调用DataLayerSetUp函数对数据层进行进一步的设置
  //这个函数除了BasePrefetchingDataLayer以外不应该被重写
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  TransformationParameter transform_param_;//数据预处理变换器参数
  shared_ptr<DataTransformer<Dtype> > data_transformer_;//数据与处理变换器
  bool output_labels_;//是否属出标签数据标志
};

template <typename Dtype>
class Batch { //包含两个Blob  分别保存 数据 与 标签
 public:
  Blob<Dtype> data_, label_;
};

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void InternalThreadEntry();//内部线程入口
  virtual void load_batch(Batch<Dtype>* batch) = 0;//载入批量数据，纯虚函数

  vector<shared_ptr<Batch<Dtype> > > prefetch_;//预取buffer
  BlockingQueue<Batch<Dtype>*> prefetch_free_;//空闲Batch队列
  BlockingQueue<Batch<Dtype>*> prefetch_full_;//已加载Batch队列
  Batch<Dtype>* prefetch_current_;

  Blob<Dtype> transformed_data_;//变换后的数据
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
