#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {//sigmoid函数
  return 0.5 * tanh(0.5 * x) + 0.5;
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();//只读获得上一层的data位置
  Dtype* top_data = top[0]->mutable_cpu_data();//读写获得下一层的data位置
  const int count = bottom[0]->count();//获得下层的运算总数
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);//对每个data进行非线性运算
  }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {//查询是否需要反向传播
    const Dtype* top_data = top[0]->cpu_data();//只读获取后一层的data
    const Dtype* top_diff = top[0]->cpu_diff();//只读获取后一层的diff梯度
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();//获取前一层的梯度地址
    const int count = bottom[0]->count();//获取前一层的计算总数
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);//链式法则梯度求解
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe
