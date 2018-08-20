#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class mutex; }

namespace caffe {

/**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with
 * their output Blob%s.
 */
template <typename Dtype>
class Layer {
 public:
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   */
  explicit Layer(const LayerParameter& param)
    : layer_param_(param) {
      // Set phase and copy blobs (if there are any).
      phase_ = param.phase();//设置当前阶段 训练/预测
      if (layer_param_.blobs_size() > 0) {//如果layer包含blob
        blobs_.resize(layer_param_.blobs_size());
        for (int i = 0; i < layer_param_.blobs_size(); ++i) { //按照layer_param的设定调整blobs
          blobs_[i].reset(new Blob<Dtype>());//申请新的blobs内存
          blobs_[i]->FromProto(layer_param_.blobs(i));//从硬盘中读取数据
        }
      }
    }
  virtual ~Layer() {}

  /**
   * @brief Implements common layer setup functionality.
   * 对一般层的设置函数
   *
   * @param bottom the preshaped input blobs
   * 上层已经设置好形状的blobs数据
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape
   *     一个已经申请好内存但是没有形状的输出blobs，将被通过Reshape函数定义形状
   *
   * Checks that the number of bottom and top blobs is correct.
   * Calls LayerSetUp to do special layer setup for individual layer types,
   * 调用LayerSetUp对每个层属性进行进一步设置
   * followed by Reshape to set up sizes of top blobs and internal buffers.
   * Sets up the loss weight multiplier blobs for any non-zero loss weights.
   * This method may not be overridden.
   * 一般这个方法不应该被重写
   */
  void SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CheckBlobCounts(bottom, top);//检查blobs
    LayerSetUp(bottom, top);//与层类型相关的配置过程
    Reshape(bottom, top);//对top Blob变形
    SetLossWeights(top);//设置损失权值因子
  }

  /**
   * @brief Does layer-specific setup: your layer should implement this function
   *        as well as Reshape.
   *        对每个层的特别设置：每个层的形状在进行重设的时候需要调用这个函数
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for
   *     this layer
   * @param top
   *     the allocated but unshaped output blobspplication-specific
   *
   * This method should do one-time layer specific setup. This includes reading
   * and processing relevent parameters from the <code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top blob sizes.
   * 这个方法在层的特别设置中调用一次。这个函数包含了从 layer_param_中读取与处理相关数据，
   * 设置顶层与内部缓存的形状应该在Reshape中完成。这应该在调用前向传播之前完成从而取调整
   * 顶层的blob大小。
   */
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  /**
   * @brief Adjust the shapes of top blobs and internal buffers to accommodate
   *        the shapes of the bottom blobs.
   *        调整顶层的blobs与内部缓存形状去适应底层pplication-specific的blobs
   *
   * @param bottom the input blobs, with the requested input shapes
   * 底层（上一层）的包含形状的blobs
   * @param top the top blobs, which should be reshaped as needed
   * 顶层（下一层）的需要被调整形状的blobs
   *
   * This method should reshape top blobs as needed according to the shapes
   * of the bottom (input) blobs, as well as reshaping any internal buffers
   * and making any other necessary adjustments so that the layer can
   * accommodate the bottom blobs.
   * 这个函数会重新设置顶层（下一层）的形状来符合下一层的要求，同时重新设定所有内部缓存
   * 或者其他必要的空间来匹配底层（上一层）的形状。
   */
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;

  /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   * 给出底层blobs，计算顶层blobs和loss（损失）
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   *     底层blobs，其内部数据空间保存了本层需要的输入数据
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     outputs
   *     顶层blobs，内部数据空间将会保存本层的输出数据
   * \return The total loss from the layer.
   * 返回 本层的总数出
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   * 前向传播装饰器调用相应的设备装饰函数(Forward_cpu or Forward_gpu)利用底层数据进行对顶层
   * 的计算。如果这层拥有非零的损失权重，装饰器将会计算并返回本层损失
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
   * 你的层应该调用 Forward_cpu 或者 (optionally) Forward_gpu惊醒前向传播运算
   */
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Given the top blob error gradients, compute the bottom blob error
   *        gradients.
   *        通过给出的上层损失梯度，计算下层的损失梯度
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error
   *     with respect to themselves
   *     输出blobs，其diff空间存储了相应的误差梯度
   * @param propagate_down
   *     a vector with equal length to bottom, with each index indicating
   *     whether to propagate the error gradients down to the bottom blob at
   *     the corresponding index
   *     一个与底层长度相等的vector，其中每个索引位置指示其是否向相应的底层元素传递了损失梯度。
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error
   *     with respect to themselves after Backward is run
   *     输入blobs 其diff空间将要存储相应的误差梯度在该函数调用后
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   *本函数也是一个反向传播修饰器，其会调用相应设备的反向传播函数(Backward_cpu or Backward_gpu)利用底层
   * 的误差数据计算顶层的误差数据    ？？？？？不应该是反向传播吗？ 这里注解应该有错误
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
   */
  inline void Backward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  /**
   * @brief Returns the vector of learnable parameter blobs.
   * 返回layer内部可训练的权值与偏置向量
   */
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

  /**
   * @brief Returns the layer parameter.
   * 返回层的参数
   */
  const LayerParameter& layer_param() const { return layer_param_; }

  /**
   * @brief Writes the layer parameter to a protocol buffer
   * 将层的参数写入到protocol缓冲区
   */
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

  /**
   * @brief Returns the scalar loss associated with a top blob at a given index.
   * 返回一个与某个toplayer相关的标量loss值
   */
  inline Dtype loss(const int top_index) const {
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
  }

  /**
   * @brief Sets the loss associated with a top blob at a given index.
   * 设置某个相关top层的loss值
   */
  inline void set_loss(const int top_index, const Dtype value) {
    if (loss_.size() <= top_index) {
      loss_.resize(top_index + 1, Dtype(0));
    }
    loss_[top_index] = value;
  }

  /**
   * @brief Returns the layer type.
   */
  virtual inline const char* type() const { return ""; }

  /**
   * @brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of bottom blobs.
   */
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
   */
  virtual inline int MinBottomBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
  virtual inline int MaxBottomBlobs() const { return -1; }
  /**
   * @brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
  virtual inline int ExactNumTopBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
  virtual inline int MinTopBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
  virtual inline int MaxTopBlobs() const { return -1; }
  /**
   * @brief Returns true if the layer requires an equal number of bottom and
   *        top blobs.
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

  /**
   * @brief Return whether "anonymous" top blobs are created automatically
   *        by the layer.
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
  virtual inline bool AutoTopBlobs() const { return false; }layers_

  /**
   * @brief Return whether to allow force_backward for a given bottom blob
   *        index.
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to blob i only if it needs gradient information
   * (as is done when force_backward == false).
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  /**
   * @brief Specifies whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *       查看该层是否计算相对应的权值或者偏置项梯度，具体相对于谁由param_id指定
   *
   * You can safely ignore false values and always compute gradients
   * for all parameters, but possibly with wasteful computation.
   */
  inline bool param_propagate_down(const int param_id) {
    return (param_propagate_down_.size() > param_id) ?
        param_propagate_down_[param_id] : false;
  }
  /**
   * @brief Sets whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *        设置该层是否计算相对应的权值或者偏置项梯度，具体相对于谁由param_id指定
   */
  inline void set_param_propagate_down(const int param_id, const bool value) {
    if (param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }


 protected:
  /** The protobuf that stores the layer parameters */
  LayerParameter layer_param_; //保存layer参数的ProtoBuffer对象
  /** The phase: TRAIN or TEST */
  Phase phase_;//当前阶段
  /** The vector that stores the learnable parameters as a set of blobs. */
  vector<shared_ptr<Blob<Dtype> > > blobs_;//内部权值与偏置项 blobs
  /** Vector indicating whether to compute the diff of each param blob. */
  vector<bool> param_propagate_down_;//标志位 是否计算对应的参数误差梯度

  /** The vector that indicates whether each top blob has a non-zero weight in
   *  the objective function. */
  vector<Dtype> loss_;//标志位，在目标函数中，是否每个top blob都有非零权重

  /** @brief Using the CPU device, compute the layer output. */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Usinglayers_ CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;
  /**
   * @brief Using the GPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
   */
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom);
  }

  /**
   * Called by the parent Layer's SetUp to check that the number of bottom
   * and top Blobs provided as input match the expected numbers specified by
   * the {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
   */
  virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
    if (ExactNumBottomBlobs() >= 0) {
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
          << type() << " Layer takes " << ExactNumBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
      CHECK_LE(MinBottomBlobs(), bottom.size())
          << type() << " Layer takes at least " << MinBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
      CHECK_GE(MaxBottomBlobs(), bottom.size())
          << type() << " Layer takes at most " << MaxBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
      CHECK_EQ(ExactNumTopBlobs(), top.size())
          << type() << " Layer produces " << ExactNumTopBlobs()
          << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
      CHECK_LE(MinTopBlobs(), top.size())
          << type() << " Layer produces at least " << MinTopBlobs()
          << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
      CHECK_GE(MaxTopBlobs(), top.size())
          << type() << " Layer produces at most " << MaxTopBlobs()
          << " top blob(s) as output.";
    }
    if (EqualNumBottomTopBlobs()) {
      CHECK_EQ(bottom.size(), top.size())
          << type() << " Layer produces one top blob as output for each "
          << "bottom blob input.";
    }
  }

  /**
   * Called by SetUp to initialize the weights associated with any top blobs in
   * the loss function. Store non-zero loss weights in the diff blob.
   * 该函数在layer的SetUp（）中调用，主要目的是初始化与Top Blob相关的loss权重，放到Top Blob
   * 的diff域，实际由forward（）函数计算
   * loss_weight ==0 表示当前层不参与loss的计算 大部分layer属于该类
   * loss_weight ==1 标识当前层参与loss计算 损失层（losslayer）属于这一类
   */
  inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {
    const int num_loss_weights = layer_param_.loss_weight_size();
    if (num_loss_weights) {
      CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
          "unspecified or specified once per top blob.";
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        const Dtype loss_weight = layer_param_.loss_weight(top_id);
        if (loss_weight == Dtype(0)) { continue; }
        this->set_loss(top_id, loss_weight);
        const int count = top[top_id]->count();
        Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
        caffe_set(count, loss_weight, loss_multiplier);
      }
    }
  }

 private:
  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype loss = 0;
  Reshape(bottom, top);
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Forward_cpu(bottom, top);
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; } //如果该层参与了损失计算 其中loss（）函数返回相应的loos计算参与标志位
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->cpu_data();//如果为LossLayer 则通过Forward函数计算出全局损失函数，并放在Top Blob Data域
      const Dtype* loss_weights = top[top_id]->cpu_diff();//如果loss_weight不为0 则在Setlossweight函数中将loss权重放在Top Blob diff域
      loss += caffe_cpu_dot(count, data, loss_weights);//loss加权求和
    }
    break;
  case Caffe::GPU:
    Forward_gpu(bottom, top);
#ifndef CPU_ONLY
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->gpu_data();
      const Dtype* loss_weights = top[top_id]->gpu_diff();
      Dtype blob_loss = 0;
      caffe_gpu_dot(count, data, loss_weights, &blob_loss);
      loss += blob_loss;
    }
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
  return loss;//返回标量loss
}

template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Backward_cpu(top, propagate_down, bottom);
    break;
  case Caffe::GPU:
    Backward_gpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

// Serialize LayerParameter to protocol buffer
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
}

}  // namespace caffe

#endif  // CAFFE_LAYER_H_
