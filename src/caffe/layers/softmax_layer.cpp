#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());//标准化索引值 通常为1
  top[0]->ReshapeLike(*bottom[0]);//按照输入数组重新映射输出数组
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();//分配空间 获得空间地址
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);//初始化求和器数据空间为 1
  outer_num_ = bottom[0]->count(0, softmax_axis_);//计算从0到softmax_axis轴的数据大小  n * c
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);//计算从softmax_axis+1到数据轴最后的数据大小 h * w
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();//临时变量
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_; //理解为标签种类
  caffe_copy(bottom[0]->count(), bottom_data, top_data);//
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
    //按照维度找最大值
  for (int i = 0; i < outer_num_; ++i) { //对每一批次进行计算
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction 输出缓冲区减去最大值
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation 求exp
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp 求1+exp(a_k)
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division 求softmax值 除法
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}

    //反向传播具体实现运算为 top_diff-(top_diff 点乘 top_data)*top_data

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {//对每个批次进行计算 通常为1
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
      //计算top_diff top_data的点积 然后从bottom_diff中减去该值
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace caffe
