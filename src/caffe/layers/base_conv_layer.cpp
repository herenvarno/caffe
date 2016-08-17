#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
....

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

////////////////////////////////////////////////////////////////////////////////
// MODIFICATION BEGIN
////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm_mod(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {

	// IF SET<=0 USE ORIGINAL CAFFE CODE
	if(this->set_<=0){
		this->forward_gpu_gemm(input, weights, output, skip_im2col);
		return;
	}

	const Dtype* col_buff = input;
	int size=0;
	int set_size = (this->conv_out_spatial_dim_+this->set_-1)/this->set_;
	for(int set_idx=0; set_idx<(conv_out_spatial_dim_+set_size-1)/set_size; set_idx++)
	{
		if (!is_1x1_) {
			if (!skip_im2col) {
				size = im2col_gpu_mod(input, conv_in_channels_,
				conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
				kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
				pad_.cpu_data()[0], pad_.cpu_data()[1],
				stride_.cpu_data()[0], stride_.cpu_data()[1],
				dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buffer_.mutable_gpu_data(), set_idx, set_size);
			}
			col_buff = col_buffer_.gpu_data();
			caffe_gpu_gemm_mod<Dtype>(CblasNoTrans, CblasNoTrans, 
				conv_out_channels_, size, kernel_dim_,  
				(Dtype)1., weights, col_buff, 
				(Dtype)0., output + set_idx*set_size, conv_out_spatial_dim_);
		}
	}
}
////////////////////////////////////////////////////////////////////////////////
// MODIFICATION END
////////////////////////////////////////////////////////////////////////////////

....

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
