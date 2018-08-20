#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Net {
 public:
  explicit Net(const NetParameter& param);
  explicit Net(const string& param_file, Phase phase,
      const int level = 0, const vector<string>* stages = NULL);
  virtual ~Net() {}

  /// @brief Initialize a network with a NetParameter.
  //使用NetParameter初始化Net
  void Init(const NetParameter& param);

  /**
   * @brief Run Forward and return the result.
   *运行向前传播 返回结果
   */
  const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);
  /// @brief DEPRECATED; use Forward() instead.
  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL) {
    LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: ForwardPrefilled() "
        << "will be removed in a future version. Use Forward().";
    return Forward(loss);
  }

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   * 形式 和 被网络采取向前与向后传播操作的变体 是特别的。对一般DAG网络而言，我们应该注意到
   * 1 从一个层到另一个层的计算也会会导致对不相关分支的计算；2 计算如果从中间开始而未包含所有网络
   * 可能会导致错误的结果。
   */
  Dtype ForwardFromTo(int start, int end);
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);
  /// @brief DEPRECATED; set input blobs then use Forward() instead.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL);

  /**
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   *        清零所有的diff域，应该在反向传播之前调用
   */
  void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   * 网络的反向传播不应该指定输入与输出，由于在前向传播过程中已经提供了联系
   */
  void Backward();
  void BackwardFromTo(int start, int end);
  void BackwardFrom(int start);
  void BackwardTo(int end);

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   * 这在不经过运行前向传播而改变层的尺寸中非常有用，例如计算输出特征的尺寸
   */
  void Reshape();

  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);
    Backward();
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
  ///根据计算好的diff值去更新网络的权重
  void Update();
  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *一个拥有权重的blobs分享权重给被分享的blobs
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   * 这个函数在Net：：Inite中被调用，因此一般情况下该函数不应被手动调用
   */
  void ShareWeights();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   *        将已经训练好的网络权值浅拷贝给一个已经初始化的网络
   */
  void ShareTrainedLayersWith(const Net* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string trained_filename);
  void CopyTrainedLayersFromHDF5(const string trained_filename);
  /// @brief Writes the net to a proto.
    ///序列化网络到proto
  void ToProto(NetParameter* param, bool write_diff = false) const;
  /// @brief Writes the net to an HDF5 file.
    ///序列化一个网络到HDF5文件
  void ToHDF5(const string& filename, bool write_diff = false) const;

  /// @brief returns the network name.
  inline const string& name() const { return name_; }
  /// @brief returns the layer names
  inline const vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the blob names
  inline const vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the blobs
  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const {
    return blobs_;
  }
  /// @brief returns the layers
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
    return layers_;
  }
  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const { return phase_; }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
    return top_vecs_;
  }
  /// @brief returns the ids of the top blobs of layer i
  inline const vector<int> & top_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
    return top_id_vecs_[i];
  }
  /// @brief returns the ids of the bottom blobs of layer i
  inline const vector<int> & bottom_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }
  inline const vector<vector<bool> >& bottom_need_backward() const {
    return bottom_need_backward_;
  }
  inline const vector<Dtype>& blob_loss_weights() const {
    return blob_loss_weights_;
  }
  inline const vector<bool>& layer_need_backward() const {
    return layer_need_backward_;
  }
  /// @brief returns the parameters
    //返回所有权值
  inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
    return params_;
  }
    //返回所有可以训练的权值
  inline const vector<Blob<Dtype>*>& learnable_params() const {
    return learnable_params_;
  }
  /// @brief returns the learnable parameter learning rate multipliers
    ///返回学习速率倍乘因子
  inline const vector<float>& params_lr() const { return params_lr_; }
  inline const vector<bool>& has_params_lr() const { return has_params_lr_; }
  /// @brief returns the learnable parameter decay multipliers
    ///返回可训练权值衰减因子
  inline const vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  inline const vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
    //返回layer名称与向量下标映射对
  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
    //返回权值所有者
  inline const vector<int>& param_owners() const { return param_owners_; }
  inline const vector<string>& param_display_names() const {
    return param_display_names_;
  }
  /// @brief Input and output blob numbers
    //返回输入输出blob数目
  inline int num_inputs() const { return net_input_blobs_.size(); }
  inline int num_outputs() const { return net_output_blobs_.size(); }
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  inline const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  inline const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  bool has_blob(const string& blob_name) const;
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
  bool has_layer(const string& layer_name) const;
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;

  void set_debug_info(const bool value) { debug_info_ = value; }

  // Helpers for Init.
  /**
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   *        layers过滤器
   */
  static void FilterNet(const NetParameter& param,
      NetParameter* param_filtered);
  /// @brief return whether NetState state meets NetStateRule rule
    //判断网络是否满足网络规则
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
      const string& layer_name);

  // Invoked at specific points during an iteration
  class Callback {
   protected:
    virtual void run(int layer) = 0;

    template <typename T>
    friend class Net;
  };
  const vector<Callback*>& before_forward() const { return before_forward_; }
  void add_before_forward(Callback* value) {
    before_forward_.push_back(value);
  }
  const vector<Callback*>& after_forward() const { return after_forward_; }
  void add_after_forward(Callback* value) {
    after_forward_.push_back(value);
  }
  const vector<Callback*>& before_backward() const { return before_backward_; }
  void add_before_backward(Callback* value) {
    before_backward_.push_back(value);
  }
  const vector<Callback*>& after_backward() const { return after_backward_; }
  void add_after_backward(Callback* value) {
    after_backward_.push_back(value);
  }

 protected:
  // Helpers for Init.
  /// @brief Append a new top blob to the net.//为网络追加一个top
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.//为网络追加一个bottom
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.//为网络追加一个权值Blob
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int param_id);

  /// @brief The network name 网络名称
  string name_;
  /// @brief The phase: TRAIN or TEST 当前阶段
  Phase phase_;
  /// @brief Individual layers in the net 网络中的独立层
  vector<shared_ptr<Layer<Dtype> > > layers_;
  vector<string> layer_names_; //层名称
  map<string, int> layer_names_index_; //层名称索引表
  vector<bool> layer_need_backward_; //层是否需要BP标志
  /// @brief the blobs storing intermediate results between the layer.
  vector<shared_ptr<Blob<Dtype> > > blobs_;//层与层之间传递数据的管道
  vector<string> blob_names_;//Blob名称
  map<string, int> blob_names_index_;//Blob名称与索引映射表
  vector<bool> blob_need_backward_;//标记某个Blob是否需要BP
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  ///bottom_vec_存放每个层的输入Blob，实际上他并不是这些Blob的所有者（blobs_才是），
  /// 他只是保存指针
  vector<vector<Blob<Dtype>*> > bottom_vecs_;//网络所有的bottom Blobs
  vector<vector<int> > bottom_id_vecs_;//bottom blobs的索引（该数组为2维结构，其用botoom_vecs中blob的索引存储了网络bottom blob的结构）
  vector<vector<bool> > bottom_need_backward_;//bottom blob是否需要反向传播的标志
  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<Dtype>*> > top_vecs_;//网络所有的top blob
  vector<vector<int> > top_id_vecs_;//top blob的索引值（该数组为2维结构，其用top_vecs中blob的索引存储了网络top blob的结构）
  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.

  vector<Dtype> blob_loss_weights_; //每个Blob对全局损失函数的投票权重 损失层为1 其他层为0
  vector<vector<int> > param_id_vecs_; //权值Blob的索引
  vector<int> param_owners_;//权值的所有者
  vector<string> param_display_names_;//权值Blob的名称
  vector<pair<int, int> > param_layer_indices_;
  map<string, int> param_names_index_;//权值名称与索引的map表
  /// blob indices for the input and the output of the net
  vector<int> net_input_blob_indices_;//网络input blob的索引值向量
  vector<int> net_output_blob_indices_;//网络output blob的索引值向量
  vector<Blob<Dtype>*> net_input_blobs_;//网络输入 Blob
  vector<Blob<Dtype>*> net_output_blobs_;//网络输出 Blob
  /// The parameters in the network.
  vector<shared_ptr<Blob<Dtype> > > params_;//权值Blob 用于存储网络权值
  vector<Blob<Dtype>*> learnable_params_;//网络可用于学习的参数
  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  vector<int> learnable_param_ids_;//存储了params_ -> learnable_params_的索引映射
  /// the learning rate multipliers for learnable_params_
  vector<float> params_lr_;//learnable_params_中每个元素的倍乘因子
  vector<bool> has_params_lr_;//是否拥有学习倍乘因子
  /// the weight decay multipliers for learnable_params_
  vector<float> params_weight_decay_;//学习衰减系数
  vector<bool> has_params_decay_;//是否拥有学习衰减系数
  /// The bytes of memory used by this net
  size_t memory_used_;//内存使用总数
  /// Whether to compute and display debug info for the net.
  bool debug_info_;//调试信息
  // Callbacks
  vector<Callback*> before_forward_;
  vector<Callback*> after_forward_;
  vector<Callback*> before_backward_;
  vector<Callback*> after_backward_;

DISABLE_COPY_AND_ASSIGN(Net);
};


}  // namespace caffe

#endif  // CAFFE_NET_HPP_
