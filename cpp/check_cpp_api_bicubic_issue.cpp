#include <iostream>
#include <cassert>

#include <torch/torch.h>
#include <torch/script.h>


namespace F = torch::nn::functional;


int main() {

  auto input = torch::arange(3 * 225 * 225, torch::CPU(torch::kFloat)).reshape({1, 3, 225, 225});

  input = (input % 50) + 100;
  // input = input % 256;

  std::cout << "input: " << input.sizes() << std::endl;
  std::cout << "input min/max: " << input.min() << ", " << input.max() << std::endl;

  auto mode = torch::kBicubic;
  // auto mode = torch::kBilinear;

  auto options = F::InterpolateFuncOptions()
                    .size(std::vector<int64_t>({128, 128}))
                    .mode(mode)
                    .antialias(true)
                    .align_corners(false);

  auto output = F::interpolate(input, options);

  output = output.round().to(torch::CPU(torch::kByte));

  std::cout << "output: " << output.sizes() << std::endl;
  std::cout << "- mean: " << output.sum() / (3.0 * 128.0 * 128.0) << std::endl;
  std::cout << "- mean R: " << output[0][0].sum() / (128.0 * 128.0) << std::endl;
  std::cout << "- mean G: " << output[0][1].sum() / (128.0 * 128.0) << std::endl;
  std::cout << "- mean B: " << output[0][2].sum() / (128.0 * 128.0) << std::endl;
  std::cout << "-----" << std::endl;
  std::cout << "- data R: \n" << output.index({0, 0, at::indexing::Slice(0, 7), at::indexing::Slice(0, 7)}) << std::endl;
  std::cout << "- data G: \n" << output.index({0, 1, at::indexing::Slice(0, 7), at::indexing::Slice(0, 7)}) << std::endl;
  std::cout << "- data B: \n" << output.index({0, 2, at::indexing::Slice(0, 7), at::indexing::Slice(0, 7)}) << std::endl;


  // https://discuss.pytorch.org/t/load-tensors-saved-in-python-from-c-and-vice-versa/39435/9
  torch::Tensor expected_input, expected_output_bicubic, expected_output_bilinear;
  try {
      torch::jit::script::Module tensors = torch::jit::load("expected_input_output_from_pillow.pt");
      c10::IValue iv = tensors.attr("input");
      c10::IValue ov1 = tensors.attr("output_bicubic");
      c10::IValue ov2 = tensors.attr("output_bilinear");
      expected_input = iv.toTensor();
      expected_output_bicubic = ov1.toTensor();
      expected_output_bilinear = ov2.toTensor();
  } catch (const c10::Error& e) {
      std::cerr << "error loading the tensors " << std::endl;
      std::cerr << e.msg() << std::endl;
  }

  output = output.to(torch::CPU(torch::kFloat));

  std::cout << "Check output vs expected by Pillow" << std::endl;
  std::cout << "- torch::allclose(expected_input, input): " << (torch::allclose(expected_input, input) ? "true" : "false") << std::endl;

  assert(torch::allclose(expected_input, input));

  std::cout << "- torch::allclose(expected_output_bicubic, output, 0, 1): " << (torch::allclose(expected_output_bicubic, output, 0, 1) ? "true" : "false") << std::endl;
  std::cout << "- torch::allclose(expected_output_bilinear, output, 0, 1): " << (torch::allclose(expected_output_bilinear, output, 0, 1) ? "true" : "false") << std::endl;

  if (!torch::allclose(expected_output_bicubic, output)) {
    auto abs_diff = (expected_output_bicubic - output).abs();
    std::cout << "Bicubic, Max Abs Error: " << abs_diff.max() << std::endl;
    auto n = (abs_diff > 0).sum();
    auto n1 = (abs_diff == 1).sum();
    auto n2 = (abs_diff == 2).sum();
    auto n3 = (abs_diff > 2).sum();
    std::cout << "Bicubic, Number of non-equal values: " << n << std::endl;
    std::cout << "Bicubic, Number of absdiff==1 values: " << n1 << std::endl;
    std::cout << "Bicubic, Number of absdiff==2 values: " << n2 << std::endl;
    std::cout << "Bicubic, Number of absdiff>2 values: " << n3 << std::endl;
  }

  if (!torch::allclose(expected_output_bilinear, output)) {
    auto abs_diff = (expected_output_bilinear - output).abs();
    std::cout << "Bilinear, Max Abs Error: " << abs_diff.max() << std::endl;
    auto n = (abs_diff > 0).sum();
    auto n1 = (abs_diff == 1).sum();
    auto n2 = (abs_diff == 2).sum();
    auto n3 = (abs_diff > 2).sum();
    std::cout << "Bilinear, Number of non-equal values: " << n << std::endl;
    std::cout << "Bilinear, Number of absdiff==1 values: " << n1 << std::endl;
    std::cout << "Bilinear, Number of absdiff==2 values: " << n2 << std::endl;
    std::cout << "Bilinear, Number of absdiff>2 values: " << n3 << std::endl;
  }

  return 0;
}