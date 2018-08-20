#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}
/*im2col_cpu将c个通道的卷积层输入图像转化为c个通道的矩阵，矩阵的行值为卷积核高*卷积核宽，
也就是说，矩阵的单列表征了卷积核操作一次处理的小窗口图像信息；而矩阵的列值为卷积层
输出单通道图像高*卷积层输出单通道图像宽，表示一共要处理多少个小窗口。
im2col_cpu接收13个参数，分别为输入数据指针(data_im)，卷积操作处理的一个卷积组的通道
数(channels)，输入图像的高(height)与宽(width)，原始卷积核的高(kernel_h)与宽(kernel_w)，
输入图像高(pad_h)与宽(pad_w)方向的pad，卷积操作高(stride_h)与宽(stride_w)方向的步长，
卷积核高(dilation_h)与宽(dilation_w)方向的扩展，输出矩阵数据指针(data_col)*/

    //图像在blob中存储的方式是 通道优先-行优先-列
    //既 同一通道的放在一起 同一行的放在一起 最后放列
    //第一通道[第一行（x x x x）第二行（x x x x ）...]第二通道[第一行(...)第二行(...)]...

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,//输入数据的指针 卷及操作中一个卷积核的通道数
    const int height, const int width, const int kernel_h, const int kernel_w,//原图的高 宽 卷积核的 高 宽
    const int pad_h, const int pad_w,//图像高与宽的pad
    const int stride_h, const int stride_w,//卷及操作 高与宽 方向的步长
    const int dilation_h, const int dilation_w,//卷积核 高与宽 方向的扩展
    Dtype* data_col) {//输出数据的指针
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;//计算卷积层输入单通道图像的数据容量
  for (int channel = channels; channel--; data_im += channel_size) {//按照通道进行处理 每次循环 data_dim移动一个通道的偏移
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {//按照核的高度循环
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {//按照核的宽度循环
        int input_row = -pad_h + kernel_row * dilation_h;//在这里找到卷积核中的某一行在输入图像中的第一个操作区域的行索引
        for (int output_rows = output_h; output_rows; output_rows--) {//按照输出高度循环
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {//判断输入行数小于零或者大于原图的高
            for (int output_cols = output_w; output_cols; output_cols--) {//说明inputrow在pad上
              *(data_col++) = 0;//在该数据上填写（输出列数）个0
            }
          } else {//input_rows不再pad上
            int input_col = -pad_w + kernel_col * dilation_w;//找到卷积核中的某一列在输入图像上的第一个操作区域的列索引
            for (int output_col = output_w; output_col; output_col--) {//在列上循环
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {//判断输入列是否大于等于0且小于原图的宽
                *(data_col++) = data_im[input_row * width + input_col];//找到对应的数据并填充data_col
              } else {//输出列在pad上
                *(data_col++) = 0;
              }
              input_col += stride_w;//输出列加上步进值
            }//输出列循环结束
          }//input_rows不在pad上 结束
          input_row += stride_h;//输出行加上步进值
        }//输出高度循环结束
      }//核宽度循环结束
    }//核高度循环结束
  }//通道循环结束
}//函数结束

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_col);

template <typename Dtype>
inline void im2col_nd_core_cpu(const Dtype* data_input, const bool im2col,
    const int num_spatial_axes, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_output) {
  if (!im2col) {
    int im_size = im_shape[0];
    for (int i = 0; i < num_spatial_axes; ++i) {
      im_size *= im_shape[1 + i];
    }
    caffe_set(im_size, Dtype(0), data_output);
  }
  int kernel_size = 1;
  for (int i = 0; i < num_spatial_axes; ++i) {//获得每个kernel块的大小
    kernel_size *= kernel_shape[i];//这里是 乘等于
  }
  const int channels_col = col_shape[0];// channels_col = inputchannel(输入图像的channel)*kernel_size
  vector<int> d_offset(num_spatial_axes, 0);// 类似于im2col中的w_offset和h_offset，只不过因为这里是n维，所以用数组表示
  vector<int> d_iter(num_spatial_axes, 0);// 类似于im2col中w和h，是col_buff中的偏移
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int offset = c_col;
    for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
      if (d_i < num_spatial_axes - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented; ) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      int index_col = c_col;
      int index_im = c_col / kernel_size;
      bool is_padding = false;
      for (int d_i = 0; d_i < num_spatial_axes; ++d_i) {
        const int d = d_iter[d_i];
        const int d_im = d * stride[d_i] - pad[d_i] +
            d_offset[d_i] * dilation[d_i];
        is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_im *= im_shape[d_i + 1];
        index_im += d_im;
      }
      if (im2col) {
        if (is_padding) {
          data_output[index_col] = 0;
        } else {
          data_output[index_col] = data_input[index_im];
        }
      } else if (!is_padding) {  // col2im
        data_output[index_im] += data_input[index_col];
      }
      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      incremented = false;
      for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
        const int d_max = col_shape[d_i + 1];
        DCHECK_LT(d_iter[d_i], d_max);
        if (d_iter[d_i] == d_max - 1) {
          d_iter[d_i] = 0;
        } else {  // d_iter[d_i] < d_max - 1
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    }  // while(incremented) {
  }  // for (int c = 0; c < channels_col; ++c) {
}

template <typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col) {
  const bool kIm2Col = true;
  im2col_nd_core_cpu(data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
                  kernel_shape, pad, stride, dilation, data_col);
}

// Explicit instantiation
template void im2col_nd_cpu<float>(const float* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_col);
template void im2col_nd_cpu<double>(const double* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_col);


    /*col2im_cpu为im2col_cpu的逆操作接收13个参数，分别为输入矩阵数据指针(data_col)，卷积操作处理的一个卷积组的通道
    数(channels)，输入图像的高(height)与宽(width)，原始卷积核的高(kernel_h)与宽(kernel_w)，
    输入图像高(pad_h)与宽(pad_w)方向的pad，卷积操作高(stride_h)与宽(stride_w)方向的步长，
    卷积核高(dilation_h)与宽(dilation_w)方向的扩展，输出图像数据指针(data_im)*/

    template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,//输入矩阵指针 一个卷积核的通道数
    const int height, const int width, const int kernel_h, const int kernel_w,//输入图像的高与宽 卷积核的高与宽
    const int pad_h, const int pad_w,//图像pad的高与宽
    const int stride_h, const int stride_w,//卷及操作的步长
    const int dilation_h, const int dilation_w,//卷积的扩展方向
    Dtype* data_im) {//输出图像指针
  caffe_set(height * width * channels, Dtype(0), data_im);//将data_im初始化为0
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;//计算输出图像的宽与高
  const int channel_size = height * width;//计算原图一个通道的数据量
  for (int channel = channels; channel--; data_im += channel_size) {//通道循环
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {//核行数循环
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {//核列数循环
        int input_row = -pad_h + kernel_row * dilation_h;//获得核某行在对应原图第一个操作区域的索引号
        for (int output_rows = output_h; output_rows; output_rows--) {//输出行数循环
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {//如果在pad区
            data_col += output_w;
          } else {//不再pad区
            int input_col = -pad_w + kernel_col * dilation_w;//换算核某列在原图上第一个操作区的索引号
            for (int output_col = output_w; output_col; output_col--) {//列数循环
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {//如果不在pad区
                data_im[input_row * width + input_col] += *data_col;//将数据映射回去  与im2col正好相反
              }
              data_col++;
              input_col += stride_w;//列数步进
            }
          }
          input_row += stride_h;//行数步进
        }
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);

template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im) {
  const bool kIm2Col = false;
  im2col_nd_core_cpu(data_col, kIm2Col, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, dilation, data_im);
}

// Explicit instantiation
template void col2im_nd_cpu<float>(const float* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_im);
template void col2im_nd_cpu<double>(const double* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_im);


}  // namespace caffe
