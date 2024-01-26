#include <iostream>

#include <ATen/ATen.h>


namespace check {

at::Tensor upsample_bilinear2d(const at::Tensor & input, at::OptionalSymIntArrayRef output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    return at::_ops::upsample_bilinear2d_vec::call(input, output_size, align_corners, scale_factors);
}

}

int main() {


    {
      auto input = at::arange(1 * 3 * 4 * 5, at::CPU(at::kInt)).reshape({1, 3, 4, 5});
      long osize[] = {2, 2};
      at::IntArrayRef oosize = osize;
      at::OptionalIntArrayRef output_size = c10::make_optional(oosize);

      at::Tensor (*Func)(const at::Tensor&, at::OptionalSymIntArrayRef, bool, c10::optional<c10::ArrayRef<double>>);
      Func = at::upsample_bilinear2d_symint;
      auto fn2 = Func;

      auto sym_output_size = c10::make_optional(c10::fromIntArrayRefSlow(*output_size));
      auto output = fn2(input, sym_output_size, false, c10::nullopt);

      std::cout << "Output: " << output.sizes() << std::endl;
    }

    // {
    //   auto input = at::arange(1 * 3 * 4 * 5, at::CPU(at::kInt)).reshape({1, 3, 4, 5});
    //   long osize[] = {2, 2};
    //   at::OptionalIntArrayRef output_size = osize;

    //   auto fn = at::native::upsample_bilinear2d;
    //   at::Tensor (*Func)(const at::Tensor&, at::OptionalIntArrayRef, bool, c10::optional<c10::ArrayRef<double>>);
    //   auto fn2 = static_cast<decltype(Func)>(fn);

    //   auto output = fn2(input, output_size, false, c10::nullopt);

    //   std::cout << "Output: " << output.sizes() << std::endl;
    // }

    {
      auto input = at::arange(1 * 3 * 4 * 5, at::CPU(at::kInt)).reshape({1, 3, 4, 5});
      long osize[] = {2, 2};
      at::OptionalIntArrayRef output_size = osize;

      auto fn = at::native::upsample_bilinear2d;
      at::Tensor (*Func)(const at::Tensor&, at::OptionalIntArrayRef, bool, c10::optional<c10::ArrayRef<double>>);
      auto fn2 = static_cast<decltype(Func)>(fn);

      auto output = fn2(input, output_size, false, c10::nullopt);

      std::cout << "Output: " << output.sizes() << std::endl;
    }


    {
      auto input = at::arange(1 * 3 * 4 * 5, at::CPU(at::kInt)).reshape({1, 3, 4, 5});
      long osize[] = {2, 2};
      at::IntArrayRef oosize = osize;
      at::OptionalIntArrayRef output_size = c10::make_optional(oosize);
      auto sym_output_size = c10::make_optional(c10::fromIntArrayRefSlow(*output_size));

      auto fn = check::upsample_bilinear2d;
      at::Tensor (*Func)(const at::Tensor&, at::OptionalSymIntArrayRef, bool, c10::optional<c10::ArrayRef<double>>);
      auto fn2 = static_cast<decltype(Func)>(fn);

      auto output = fn2(input, sym_output_size, false, c10::nullopt);

      std::cout << "Output: " << output.sizes() << std::endl;
    }

    return 0;
}