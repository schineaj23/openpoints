/*
batch version of ball query, modified from the original implementation of official PointNet++ codes.
Written by PointNeXt team 
All Rights Reserved 2022.
*/

#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "ballquery_cuda_kernel.h"


#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int ballquery_cuda(int m, float radius, int nsample,
                   at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                   at::Tensor offset_tensor, at::Tensor new_offset_tensor,
                   at::Tensor idx_tensor)
{
  // CHECK_INPUT(new_xyz_tensor);
  // CHECK_INPUT(xyz_tensor);
  const float *new_xyz = new_xyz_tensor.data_ptr<float>();
  const float *xyz = xyz_tensor.data_ptr<float>();
  const int *offset = offset_tensor.data_ptr<int>();
  const int *new_offset = new_offset_tensor.data_ptr<int>();
  int *idx = idx_tensor.data_ptr<int>();

  ballquery_launcher(m, radius, nsample, xyz, new_xyz, offset, new_offset, idx);
  return 1;
}
