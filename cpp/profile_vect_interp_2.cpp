#include <iostream>
#include <chrono>

// Torch
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/native/UpSample.h>

#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/intrinsics.h>
#include <c10/util/irange.h>

#include <ATen/ops/empty.h>


namespace test {

using namespace at;
using namespace at::native;

// Helper structs to use with upsample_generic_Nd_kernel_impl
struct HelperInterpBase {

  static inline void init_indices_weights(
    at::ScalarType output_type,
    std::vector<Tensor> & output, int64_t output_size, int64_t ndims,
    int64_t reshape_dim, int interp_size
  ) {

    auto new_shape = std::vector<int64_t>(ndims, 1);
    new_shape[reshape_dim] = output_size;

    for (const auto j C10_UNUSED : c10::irange(interp_size)) {
      output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
      output.emplace_back(empty(new_shape, CPU(output_type)));
    }
  }

  template <typename scalar_t, typename aa_filter_fn_t>
  static inline void _compute_weights_aa(
    const int64_t i, const int64_t input_size, const scalar_t scale, const scalar_t support,
    scalar_t* wt_ptr, const int64_t max_interp_size, aa_filter_fn_t filter_fn,
    int64_t& xmin, int64_t& xsize, bool antialias, double align_corners_delta
  ) {

    // align_corners_delta is 0.5 for uint8 and align_corners=true and antialias=false
    //                     is 0.0 otherwise
    scalar_t center = scale * (i + 0.5 - align_corners_delta);
    scalar_t total_w = 0.0;
    scalar_t invscale = (scale >= 1.0 && antialias) ? 1.0 / scale : 1.0;
    xmin = std::max(
        static_cast<int64_t>(center - support + 0.5 + align_corners_delta), static_cast<int64_t>(0));
    xsize = std::min(
        static_cast<int64_t>(center + support + 0.5 + align_corners_delta), input_size) - xmin;

    int64_t j = 0;
    for (; j < xsize; j++) {
      scalar_t w = filter_fn((j + xmin - center + 0.5 - align_corners_delta) * invscale);
      wt_ptr[j] = w;
      total_w += w;
    }
    for (j = 0; j < xsize; j++) {
      if (total_w != 0.0) {
        wt_ptr[j] /= total_w;
      }
    }
    for (; j < max_interp_size; j++) {
      wt_ptr[j] = static_cast<scalar_t>(0.0);
    }
  }

  // Note [ Support for antialias=False as a subcase of antilias=True ]
  // This function was originally written with the hard assumption that
  // antialias=True (hence the aa in the name). It was later extended to support
  // antialias=False. The only difference between aa and no-aa is in how the
  // weights and indices are computed (and their number). In aa their number is
  // variable but with no-aa, they're fixed to interp_size. The same "filters"
  // can be used otherwise. HOWEVER, support for antialias=False here may not be
  // optimally optimized: the code assumes an arbitrary number of weights and
  // indices, but this can be optimized further when aa=False since we know
  // their actual dimensions.
  template <typename scalar_t, typename aa_filter_fn_t, int weight_index_stride=sizeof(scalar_t)>
  static inline std::tuple<std::vector<Tensor>, int> _compute_indices_weights_aa(
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims,
    int64_t reshape_dim, scalar_t scale,
    int interp_size, aa_filter_fn_t aa_filter_fn, bool antialias, double align_corners_delta
  ) {

    std::vector<Tensor> output;

    scalar_t support;
    int max_interp_size;
    if (antialias) {
        support = (scale >= 1.0) ? (interp_size * 0.5) * scale : interp_size * 0.5;
        max_interp_size = (int) std::ceil(support) * 2 + 1;
    } else {
        support = interp_size * 0.5;
        max_interp_size = interp_size;
    }

    auto new_shape = std::vector<int64_t>(ndims, 1);
    new_shape[reshape_dim] = output_size;

    // Bounds approach as in PIL: xmin/xmax
    output.emplace_back(
        empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
    output.emplace_back(
        empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
    output.emplace_back(
        empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));

    {
      // Weights
      new_shape[reshape_dim] = output_size * max_interp_size;
      auto wts = empty(new_shape, CPU(c10::CppTypeToScalarType<scalar_t>()));
      auto strides = wts.strides().vec();
      strides[reshape_dim] = 0;
      new_shape[reshape_dim] = output_size;
      wts = wts.as_strided(new_shape, strides);
      output.emplace_back(wts);
      // Weights indices
      output.emplace_back(
          empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
    }

    int64_t* idx_ptr_xmin = output[0].data_ptr<int64_t>();
    int64_t* idx_ptr_size = output[1].data_ptr<int64_t>();
    int64_t* idx_ptr_stride = output[2].data_ptr<int64_t>();
    scalar_t* wt_ptr = output[3].data_ptr<scalar_t>();
    int64_t* wt_idx_ptr = output[4].data_ptr<int64_t>();

    int64_t xmin, xmax;

    for (const auto i : c10::irange(output_size)) {
      HelperInterpBase::_compute_weights_aa(
          i,
          input_size,
          scale,
          support,
          wt_ptr + i * max_interp_size,
          max_interp_size,
          aa_filter_fn,
          xmin,
          xmax,
          antialias,
          align_corners_delta);

      idx_ptr_xmin[i] = xmin * stride;
      idx_ptr_size[i] = xmax;
      idx_ptr_stride[i] = stride;
      wt_idx_ptr[i] = i * max_interp_size * weight_index_stride;
    }
    return {output, max_interp_size};
  }

  /*
  NOTE [ Weights computation for uint8_t and multiplication trick ]
  When the input/output dtype is uint8_t, we still compute the interpolation
  weights as double, but then convert them to int16 via some conversion logic
  detailed below. This allows us to compute all interpolation operation (sum of
  multiplications) as ints instead of floats. The result is converted back into
  uint8 in basic_loop_aa_horizontal<uint8_t> (and vertical)

  In essence the idea is to avoid a multiplication between a float (the
  weight) and an int (the pixel value) and instead run a multpilication between
  2 ints:

  ```py
  COEF_PREC = 16

  def mul(a:float, b:int) -> Tuple[float, int]:
    # return a * b, round(a * b)
    actual = a * b

    assert a > 0  # I'm lazy
    int_a = floor(0.5 + a * (1 << COEF_PREC))
    with_trick = ((int_a * b) + (1 << (COEF_PREC - 1))) >> COEF_PREC

    return actual, with_trick  # round(actual) == with_trick!!
  ```

  Here's how it works:
  N == COEFF_PREC
  1 << N == 2**N
  floor(0.5 + x) == round(x)

  So the operation is something like

  int_a = round(a * 2**N)  -- let's just say it's `a * 2**N` for simplicity

  res = ((int_a * b) + (1 << (N - 1))) >> N
      = ((a * 2**N * b + 2**(N - 1)) / 2**N
      = a * b + 0.5
      = round(a * b)
      = what we wanted
  */
  template <typename aa_filter_fn_t>
  static inline std::tuple<std::vector<Tensor>, int, unsigned int> _compute_indices_int16_weights_aa(
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims,
    int64_t reshape_dim, bool align_corners, const c10::optional<double> opt_scale,
    int interp_size, aa_filter_fn_t aa_filter_fn, bool antialias, bool align_i32=false
  ) {

    double scale = area_pixel_compute_scale<double>(
        input_size, output_size, align_corners, opt_scale);

    std::vector<Tensor> indices_weights;
    auto align_corners_delta = (align_corners && !antialias) ? 0.5 : 0.0;
    std::tie(indices_weights, interp_size) = HelperInterpBase::_compute_indices_weights_aa<double, aa_filter_fn_t, sizeof(int16_t)>(
        input_size, output_size, stride, ndims, reshape_dim, scale, interp_size, aa_filter_fn, antialias, align_corners_delta);

    // Rescale float weights to int16 and compute weights precision
    auto weights_f64 = indices_weights[3];
    double * data_f64 = weights_f64.data_ptr<double>();
    int64_t weights_f64_size = output_size * interp_size;
    // can't use weights_f64.max() here as tensor is restrided
    double w_max = data_f64[0];
    for (const auto i : c10::irange(weights_f64_size)) {
        double v = data_f64[i];
        if (w_max < v) {
            w_max = v;
        }
    }

    unsigned int weights_precision = 0;
    for (weights_precision = 0; weights_precision < 22; weights_precision += 1) {
        int next_value = (int) (0.5 + w_max * (1 << (weights_precision + 1)));
        if (next_value >= (1 << 15))
            break;
    }

    // Rescale float values to int16
    int16_t * data_i16 = (int16_t *) data_f64;
    auto aligned_interp_size = interp_size;

    if (align_i32) {
      // We should respect int32 alignment as we will load int16 data as int32
      // See ImagingResampleHorizontalConvolution8u4x, mmk0 = _mm256_set1_epi32(*(int32_t*)&k[x]);
      // compute aligned_interp_size = nearest pair value to interp_size
      while (aligned_interp_size % sizeof(int32_t) != 0) {
        aligned_interp_size += 1;
      }
      // assert that we wont go out of bounds
      TORCH_INTERNAL_ASSERT(aligned_interp_size * sizeof(int16_t) < interp_size * sizeof(double));
    }

    for (const auto j : c10::irange(output_size)) {
      for (const auto k : c10::irange(interp_size)) {
        double v = data_f64[j * interp_size + k];
        if (v < 0) {
            data_i16[j * aligned_interp_size + k] = (int) (-0.5 + v * (1 << weights_precision));
        } else {
            data_i16[j * aligned_interp_size + k] = (int) (0.5 + v * (1 << weights_precision));
        }
      }
    }

    return {indices_weights, aligned_interp_size, weights_precision};
  }

};


struct HelperInterpLinear : public HelperInterpBase {

  static const int interp_size = 2;

  // Compute indices and weights for each interpolated dimension
  // indices_weights = {
  //      {indices_0, weights_0, indices_1, weights_1},  // dim -n
  //      {indices_0, weights_0, indices_1, weights_1},  // dim -(n-1)
  //      ...
  //      {indices_0, weights_0, indices_1, weights_1},  // dim -1
  // }
  // Indices and weights are reshaped as (1, 1, ..., N, ..., 1, 1) to
  // fit input/output tensors.
  // Indices are already containing the strides to optimize the computations
  static inline std::vector<Tensor> compute_indices_weights(
    at::ScalarType scalar_type,
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims, int64_t reshape_dim,
    bool align_corners, const c10::optional<double> opt_scale
  ) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<Tensor> output;
    HelperInterpLinear::init_indices_weights(
      scalar_type, output, output_size, ndims, reshape_dim, HelperInterpLinear::interp_size);
    AT_DISPATCH_FLOATING_TYPES_AND(
      ScalarType::BFloat16, scalar_type, "compute_indices_weights_linear", [&] {
        scalar_t scale = area_pixel_compute_scale<scalar_t>(input_size, output_size, align_corners, opt_scale);

        auto input_index0_ptr = output[0].data_ptr<int64_t>();
        auto lambda0_ptr = output[1].data_ptr<scalar_t>();
        auto input_index1_ptr = output[2].data_ptr<int64_t>();
        auto lambda1_ptr = output[3].data_ptr<scalar_t>();

        for (const auto i : c10::irange(output_size)) {

          compute_source_index_and_lambda<scalar_t>(
            input_index0_ptr[i], input_index1_ptr[i],
            lambda0_ptr[i], lambda1_ptr[i],
            scale, i, input_size, output_size, align_corners
          );
          // put stride into indices
          // index values correspond to input indices (0, 1, 2, 3, ...)
          // when multiplied by input stride, maximum possible value
          // input_size[dim-1] * input_size[dim-2] * ... for the given dimension.
          input_index0_ptr[i] *= stride;
          input_index1_ptr[i] *= stride;
        }
      }
    );
    return output;
  }

  // taken from
  // https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
  // src/libImaging/Resample.c#L20-L29
  template<typename scalar_t>
  static inline scalar_t aa_filter(scalar_t x) {
    if (x < 0.0) {
      x = -x;
    }
    if (x < 1.0) {
      return 1.0 - x;
    }
    return 0.0;
  }

  static inline std::vector<Tensor> compute_indices_weights_aa(
    at::ScalarType scalar_type,
    int64_t input_size,
    int64_t output_size,
    int64_t stride,
    int64_t ndims,
    int64_t reshape_dim,
    bool align_corners,
    const c10::optional<double> opt_scale
  ) {

    std::vector<Tensor> indices_weights;
    AT_DISPATCH_FLOATING_TYPES(
      scalar_type, "compute_indices_weights_aa", [&] {

        scalar_t scale = area_pixel_compute_scale<scalar_t>(
            input_size, output_size, align_corners, opt_scale);

        auto interp_size = HelperInterpLinear::interp_size;
        int unused;

        std::tie(indices_weights, unused) = HelperInterpLinear::_compute_indices_weights_aa<scalar_t>(
            input_size,
            output_size,
            stride,
            ndims,
            reshape_dim,
            scale,
            interp_size,
            &HelperInterpLinear::aa_filter<scalar_t>,
            /*antialias=*/true,
            /*align_corners_delta=*/0.0);
      }
    );
    return indices_weights;
  }

  static inline std::tuple<std::vector<Tensor>, int, unsigned int> compute_indices_int16_weights_aa(
    int64_t input_size,
    int64_t output_size,
    int64_t stride,
    int64_t ndims,
    int64_t reshape_dim,
    bool align_corners,
    const c10::optional<double> opt_scale,
    bool antialias,
    bool align_i32=false
  ) {

    auto interp_size = HelperInterpLinear::interp_size;
    auto fn = HelperInterpLinear::aa_filter<double>;
    return HelperInterpLinear::_compute_indices_int16_weights_aa(
        input_size, output_size, stride, ndims, reshape_dim,
        align_corners, opt_scale, interp_size, fn, antialias, align_i32);
  }
};


static __m128i inline mm_cvtepu8_epi32(const uint8_t* C10_RESTRICT ptr) {
  return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)ptr));
}

// TODO: We may want to hard-code an unrolled version for the case where
// num_channels=3 to hint the compiler to vectorize this (looks at original
// PIL-SIMD's code).
at::Tensor unpack_rgb(const at::Tensor& packed_tensor) {
  // Convert a "packed" tensor (typically RGBRGBRGB if channels_last) into
  // RGBARGBARGBA format where A is hard-coded to 255. Each pixel is encoded
  // into as 32bits. This generalizes to num_channels <= 4 and also works for
  // non-channels_last tensors.

  // std::cout << "- unpack_rgb" << std::endl;

  const uint8_t* packed = (const uint8_t*)packed_tensor.data_ptr<uint8_t>();
  auto num_pixels = packed_tensor.size(1) * packed_tensor.size(2);
  auto num_channels = packed_tensor.size(0);

  constexpr int rgba_size = 4;
  auto unpacked_tensor = at::empty({rgba_size, packed_tensor.size(1), packed_tensor.size(2)}, at::CPU(at::kByte));
  uint8_t* unpacked = (uint8_t*) unpacked_tensor.data_ptr<uint8_t>();

  auto stride_i = packed_tensor.stride(2);
  auto stride_j = packed_tensor.stride(0);

  for (const auto i : c10::irange(num_pixels)) {
    for (const auto j : c10::irange(rgba_size)) {
      unpacked[rgba_size * i + j] = (j < num_channels) ? packed[stride_i * i + stride_j * j] : 0;
    }
  }
  return unpacked_tensor;
}

void pack_rgb(
    const at::Tensor& unpacked_tensor, // IN
    const at::Tensor& packed_tensor // OUT
) {
  constexpr int rgba_size = 4;
  uint8_t* unpacked = (uint8_t*)unpacked_tensor.data_ptr<uint8_t>();
  uint8_t* packed = (uint8_t*)packed_tensor.data_ptr<uint8_t>();
  auto num_pixels = packed_tensor.size(1) * packed_tensor.size(2);
  auto num_channels = packed_tensor.size(0);

  auto packed_increment = packed_tensor.stride(2);
  auto packed_stride = packed_tensor.stride(0);

  for (const auto i C10_UNUSED : c10::irange(num_pixels)) {
    for (const auto j : c10::irange(num_channels)) {
      packed[j * packed_stride] = unpacked[j];
    }
    unpacked += rgba_size;
    packed += packed_increment;
  }
}

void ImagingResampleHorizontalConvolution8u4x(
    uint8_t* C10_RESTRICT lineOut0,
    uint8_t* C10_RESTRICT lineOut1,
    uint8_t* C10_RESTRICT lineOut2,
    uint8_t* C10_RESTRICT lineOut3,
    const uint8_t* C10_RESTRICT lineIn0,
    const uint8_t* C10_RESTRICT lineIn1,
    const uint8_t* C10_RESTRICT lineIn2,
    const uint8_t* C10_RESTRICT lineIn3,
    int64_t xsize,
    const int64_t* xbounds,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels);

void ImagingResampleHorizontalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    const int64_t* xbounds,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels);

void ImagingResampleVerticalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    int64_t ymin,
    int64_t ymax,
    const int16_t* k,
    unsigned int coefs_precision,
    int64_t num_channels);

void ImagingResampleHorizontal(
    const at::Tensor & unpacked_output,
    const at::Tensor & unpacked_input,
    int ksize,
    const std::vector<at::Tensor>& horiz_indices_weights,
    unsigned int horiz_weights_precision) {
  // TODO: we may want to merge that into the fallback code (currently called
  // basic_loop_aa_horizontal<uint8_t>)
  // Although this may not be needed if / when we port all this code to use
  // Vec.h since this would potentially give us another fall-back implem

  int16_t* kk = (int16_t*)(horiz_indices_weights[3].data_ptr<double>());

  auto xout = unpacked_output.size(2);
  auto yout = unpacked_output.size(1);
  auto xin = unpacked_input.size(2);
  auto num_channels = unpacked_input.size(0);

  TORCH_INTERNAL_ASSERT(num_channels == 3 || num_channels == 4);

  std::vector<int64_t> bounds_vec(2 * xout, 0);
  int64_t* bounds = bounds_vec.data();

  int64_t* idx_ptr_xmin = horiz_indices_weights[0].data_ptr<int64_t>();
  int64_t* idx_ptr_size = horiz_indices_weights[1].data_ptr<int64_t>();
  for (int i = 0; i < xout; i++) {
    bounds[2 * i + 0] = idx_ptr_xmin[i];
    bounds[2 * i + 1] = idx_ptr_size[i];
  }

  uint8_t* unpacked_input_p = unpacked_input.data_ptr<uint8_t>();
  uint8_t* unpacked_output_p = unpacked_output.data_ptr<uint8_t>();

  int64_t yy = 0;
  auto xout_stride = xout * num_channels;
  auto xin_stride = xin * num_channels;
  for (; yy < yout - 3; yy += 4) {
    ImagingResampleHorizontalConvolution8u4x(
        unpacked_output_p + yy * xout_stride,
        unpacked_output_p + (yy + 1) * xout_stride,
        unpacked_output_p + (yy + 2) * xout_stride,
        unpacked_output_p + (yy + 3) * xout_stride,
        unpacked_input_p + yy * xin_stride,
        unpacked_input_p + (yy + 1) * xin_stride,
        unpacked_input_p + (yy + 2) * xin_stride,
        unpacked_input_p + (yy + 3) * xin_stride,
        xout,
        bounds,
        kk,
        ksize,
        horiz_weights_precision,
        num_channels);
  }
  for (; yy < yout; yy++) {
    ImagingResampleHorizontalConvolution8u(
        unpacked_output_p + yy * xout_stride,
        unpacked_input_p + yy * xin_stride,
        xout,
        bounds,
        kk,
        ksize,
        horiz_weights_precision,
        num_channels);
  }
}

void ImagingResampleVertical(
    const at::Tensor & unpacked_output,
    const at::Tensor & unpacked_input,
    int ksize,
    const std::vector<at::Tensor>& vert_indices_weights,
    unsigned int vert_weights_precision) {
  // TODO: we may want to merge that into the fallback code (currently called
  // basic_loop_aa_vertical<uint8_t>)
  // Although this may not be needed if / when we port all this code to use
  // Vec.h since this would potentially give us another fall-back implem
  int16_t* k = nullptr;
  int16_t* kk = (int16_t*)(vert_indices_weights[3].data_ptr<double>());

  int64_t* idx_ptr_xmin = vert_indices_weights[0].data_ptr<int64_t>();
  int64_t* idx_ptr_size = vert_indices_weights[1].data_ptr<int64_t>();

  uint8_t* unpacked_output_p = unpacked_output.data_ptr<uint8_t>();
  uint8_t* unpacked_input_p = unpacked_input.data_ptr<uint8_t>();

  auto xout = unpacked_output.size(2);
  auto yout = unpacked_output.size(1);
  auto num_channels = unpacked_input.size(0);

  TORCH_INTERNAL_ASSERT(num_channels == 3 || num_channels == 4);

  auto xout_stride = xout * num_channels;
  for (const auto yy : c10::irange(yout)) {
    k = &kk[yy * ksize];

    auto ymin = idx_ptr_xmin[yy];
    auto ymax = idx_ptr_size[yy];
    ImagingResampleVerticalConvolution8u(
        unpacked_output_p + yy * xout_stride,
        unpacked_input_p,
        xout,
        ymin,
        ymax,
        k,
        vert_weights_precision,
        num_channels);
  }
}

// This is the only public entry point in this file.  It supports bilinear
// mode for uint8 dtype when C <= 4, with or without antialias. The
// implem is based on PIL-SIMD.
// Its equivalent implementation (fallback) for when AVX isn't supported or when
// C > 4 is separable_upsample_generic_Nd_kernel_impl()  There are a bunch of
// future improvement that can be done: look for the TODOs in this file.
// For details on how the weights are computed and how the multiplications are
// run on int (instead of float weights), see
// [ Weights computation for uint8_t and multiplication trick ]
// For details on how the AVX kernels are implemented, see
// https://gist.github.com/NicolasHug/47c97d731f05eaad5694c173849b86f5
// See also [ Support for antialias=False as a subcase of antilias=True ] to
// learn more about how the antialias=False case is computed. The same holds
// here: all these kernels are general enough to handle an arbitrary number of
// weights, but when aa=False they could be optimized further.
template <typename scale_type, class F>
void upsample_avx_bilinear_uint8(
    const at::Tensor& input,
    const at::Tensor& output,
    bool align_corners,
    const scale_type& scales,
    bool antialias) {
  auto batch_size = input.size(0);
  auto num_channels = input.size(1);
  auto xin = input.size(3);
  auto yin = input.size(2);
  auto xout = output.size(3);
  auto yout = output.size(2);

  if (xin == xout && yin == yout) {
    output.copy_(input);
    return;
  }

  auto need_horizontal = xout != xin;
  auto need_vertical = yout != yin;

  int ksize_horiz, ksize_vert;
  std::vector<at::Tensor> horiz_indices_weights, vert_indices_weights;
  unsigned int horiz_weights_precision, vert_weights_precision;

  if (need_horizontal) {
    // std::cout << "- Compute horizontal weights" << std::endl;

    int interp_dim = 3;
    std::tie(horiz_indices_weights, ksize_horiz, horiz_weights_precision) =
        F::compute_indices_int16_weights_aa(
            /*input_size=*/xin,
            /*output_size=*/xout,
            /*stride=*/1,
            /*ndims=*/4,
            /*reshape_dim=*/interp_dim,
            /*align_corners=*/align_corners,
            /*opt_scale=*/scales[interp_dim - 2],
            /*antialias=*/antialias,
            /*align_i32=*/true);
  }

  if (need_vertical) {
    // std::cout << "- Compute vertical weights" << std::endl;

    int interp_dim = 2;
    std::tie(vert_indices_weights, ksize_vert, vert_weights_precision) =
        F::compute_indices_int16_weights_aa(
            /*input_size=*/yin,
            /*output_size=*/yout,
            /*stride=*/1,
            /*ndims=*/4,
            /*reshape_dim=*/interp_dim,
            /*align_corners=*/align_corners,
            /*opt_scale=*/scales[interp_dim - 2],
            /*antialias=*/antialias,
            /*align_i32=*/true);
  }

  bool is_rgb_or_rgba = (num_channels == 3 || num_channels == 4) && \
      input.is_contiguous(at::MemoryFormat::ChannelsLast);

  // std::cout << "- is_rgb_or_rgba: " << ((is_rgb_or_rgba) ? "true" : "false") << std::endl;

  at::Tensor buffer_horiz, buffer_vert;
  if (need_horizontal && !(is_rgb_or_rgba && !need_vertical)) {
    // std::cout << "- Allocatate horizontal buffer" << std::endl;
    auto c = (is_rgb_or_rgba) ? num_channels : 4;
    buffer_horiz = at::empty({c, yin, xout}, input.options());
  }
  if (need_vertical && !is_rgb_or_rgba) {
    // std::cout << "- Allocatate vectical buffer" << std::endl;
    auto c = (is_rgb_or_rgba) ? num_channels : 4;
    buffer_vert = at::empty({c, yout, xout}, input.options());
  }

  // TODO: The unpack / pack operations create a copy of the original input and
  // output tensor. There should be a way to avoid these copies by instead
  // modifying the low-level kernels. Or maybe at least avoid copying the entire
  // tensors and just copy part of them (line by line).
  for (const auto i : c10::irange(batch_size)) {
    at::Tensor unpacked_input = (is_rgb_or_rgba) ? input[i] : unpack_rgb(input[i]);
    at::Tensor unpacked_output;

    if (need_horizontal) {
      at::Tensor unpacked_output_temp = (is_rgb_or_rgba && !need_vertical) ? output[i] : buffer_horiz;
      // std::cout << "- Apply ImagingResampleHorizontal" << std::endl;

      ImagingResampleHorizontal(
          unpacked_output_temp,
          unpacked_input,
          ksize_horiz,
          horiz_indices_weights,
          horiz_weights_precision);
      unpacked_output = unpacked_input = unpacked_output_temp;
    }
    if (need_vertical) {
      unpacked_output = (is_rgb_or_rgba) ? output[i] : buffer_vert;

      // std::cout << "- Apply ImagingResampleVertical" << std::endl;

      ImagingResampleVertical(
          unpacked_output,
          unpacked_input,
          ksize_vert,
          vert_indices_weights,
          vert_weights_precision);
    }

    TORCH_INTERNAL_ASSERT(unpacked_output.defined());

    if (!is_rgb_or_rgba) {
      // std::cout << "- pack_rgb" << std::endl;
      pack_rgb(unpacked_output, output[i]);
    }
  }
}

void ImagingResampleHorizontalConvolution8u4x(
    uint8_t* C10_RESTRICT lineOut0,
    uint8_t* C10_RESTRICT lineOut1,
    uint8_t* C10_RESTRICT lineOut2,
    uint8_t* C10_RESTRICT lineOut3,
    const uint8_t* C10_RESTRICT lineIn0,
    const uint8_t* C10_RESTRICT lineIn1,
    const uint8_t* C10_RESTRICT lineIn2,
    const uint8_t* C10_RESTRICT lineIn3,
    int64_t xsize,
    const int64_t* xbounds,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels) {

  const __m256i masks_low_high_c4[2] = {
    _mm256_set_epi8(
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0,
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    ),
    _mm256_set_epi8(
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8,
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8
    )
  };
  const __m256i masks_low_high_c3[2] = {
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0,
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6,
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6
    )
  };

  const auto mask_low = (num_channels == 3) ? masks_low_high_c3[0] : masks_low_high_c4[0];
  const auto mask_high = (num_channels == 3) ? masks_low_high_c3[1] : masks_low_high_c4[1];

  const auto mask_ch = _mm256_set_epi8(
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 1, 0,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 1, 0
  );

  const auto stride = num_channels * 1;  // num channels * sizeof(uint8)

  // xsize = output width, xx = output x index
  // xmax = interpolation size, x = interpolation index (horizontal <-> x dimension)
  // xmin = input x start index corresponding to output x index (xx)

  const auto zero = _mm256_setzero_si256();
  const auto initial = _mm256_set1_epi32(1 << (coefs_precision - 1));

  for (const auto xx : c10::irange(xsize)) {
    const auto xmin = xbounds[xx * 2 + 0];
    const auto xmax = xbounds[xx * 2 + 1];
    const auto k = &kk[xx * kmax];
    int64_t x = 0;

    auto sss0 = initial;
    auto sss1 = initial;

    // lineIn0 + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
    // --> x <= xmax - 16.0 / stride
    const auto b4_xmax = xmax - 16.0 / stride + 1;

    for (; x < b4_xmax; x += 4) {
      auto mmk0 = _mm256_set1_epi32(*(int32_t*)&k[x]);
      auto mmk1 = _mm256_set1_epi32(*(int32_t*)&k[x + 2]);

      auto source = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si128((__m128i *) (lineIn0 + stride * (x + xmin)))),
          _mm_loadu_si128((__m128i *) (lineIn1 + stride * (x + xmin))), 1);

      auto pix = _mm256_shuffle_epi8(source, mask_low);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk0));
      pix = _mm256_shuffle_epi8(source, mask_high);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk1));

      source = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si128((__m128i *) (lineIn2 + stride * (x + xmin)))),
          _mm_loadu_si128((__m128i *) (lineIn3 + stride * (x + xmin))), 1);
      pix = _mm256_shuffle_epi8(source, mask_low);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk0));
      pix = _mm256_shuffle_epi8(source, mask_high);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk1));
    }

    // lineIn0 + stride * (x + xmin) + 8 <= lineIn0 + stride * (xmax + xmin)
    // --> x <= xmax - 8.0 / stride
    const auto b2_xmax = xmax - 8.0 / stride + 1;

    for (; x < b2_xmax; x += 2) {
      auto mmk = _mm256_set1_epi32(*(int32_t*)&k[x]);

      auto pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si64(lineIn0 + stride * (x + xmin))),
          _mm_loadu_si64(lineIn1 + stride * (x + xmin)), 1);
      pix = _mm256_shuffle_epi8(pix, mask_low);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

      pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si64(lineIn2 + stride * (x + xmin))),
          _mm_loadu_si64(lineIn3 + stride * (x + xmin)), 1);
      pix = _mm256_shuffle_epi8(pix, mask_low);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
    }

    for (; x < xmax - 1; x++) {
      auto mmk = _mm256_set1_epi32(k[x]);
      auto pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(lineIn0 + stride * (x + xmin))),
          mm_cvtepu8_epi32(lineIn1 + stride * (x + xmin)), 1);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

      pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(lineIn2 + stride * (x + xmin))),
          mm_cvtepu8_epi32(lineIn3 + stride * (x + xmin)), 1);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
    }

    if (x == xmax - 1) {
      // last element x = xmax - 1
      auto mmk = _mm256_set1_epi32(k[x]);
      __m256i pix;
      __m128i p0, p1;
      if (num_channels == 3) {
        uint8_t output0[4];
        uint8_t output1[4];
        std::memcpy(output0, lineIn0 + stride * (x + xmin), 3);
        std::memcpy(output1, lineIn1 + stride * (x + xmin), 3);
        p0 = mm_cvtepu8_epi32(output0);
        p1 = mm_cvtepu8_epi32(output1);
      } else {
        p0 = mm_cvtepu8_epi32(lineIn0 + stride * (x + xmin));
        p1 = mm_cvtepu8_epi32(lineIn1 + stride * (x + xmin));
      }
      pix = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

      if (num_channels == 3) {
        uint8_t output0[4];
        uint8_t output1[4];
        std::memcpy(output0, lineIn2 + stride * (x + xmin), 3);
        std::memcpy(output1, lineIn3 + stride * (x + xmin), 3);
        p0 = mm_cvtepu8_epi32(output0);
        p1 = mm_cvtepu8_epi32(output1);
      } else {
        p0 = mm_cvtepu8_epi32(lineIn2 + stride * (x + xmin));
        p1 = mm_cvtepu8_epi32(lineIn3 + stride * (x + xmin));
      }
      pix = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
    }

    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    sss0 = _mm256_packs_epi32(sss0, zero);
    sss1 = _mm256_packs_epi32(sss1, zero);
    sss0 = _mm256_packus_epi16(sss0, zero);
    sss1 = _mm256_packus_epi16(sss1, zero);
    if (num_channels == 3) {
        sss0 = _mm256_shuffle_epi8(sss0, mask_ch);
        sss1 = _mm256_shuffle_epi8(sss1, mask_ch);
    }

    auto o0 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 0));
    auto o1 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 1));
    auto o2 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 0));
    auto o3 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 1));
    if (num_channels == 3 && C10_UNLIKELY(stride * xx + 4 >= xsize * stride)) {
        std::memcpy(lineOut0 + stride * xx, (unsigned char *) &o0, num_channels);
        std::memcpy(lineOut1 + stride * xx, (unsigned char *) &o1, num_channels);
        std::memcpy(lineOut2 + stride * xx, (unsigned char *) &o2, num_channels);
        std::memcpy(lineOut3 + stride * xx, (unsigned char *) &o3, num_channels);
    } else {
      *(uint32_t *)(lineOut0 + stride * xx) = o0;
      *(uint32_t *)(lineOut1 + stride * xx) = o1;
      *(uint32_t *)(lineOut2 + stride * xx) = o2;
      *(uint32_t *)(lineOut3 + stride * xx) = o3;
    }
  }
}

void ImagingResampleHorizontalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    const int64_t* xbounds,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels) {

  const auto kmask_low = _mm256_set_epi8(
      11, 10, 9, 8, 11, 10, 9, 8, 11, 10, 9, 8, 11, 10, 9, 8,
      3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);
  const auto kmask_high = _mm256_set_epi8(
      15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12,
      7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4);
  const auto kmask_hl = _mm256_set_epi8(
      7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4,
      3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);

  const auto mask_ch = _mm_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 1, 0);

  const __m256i masks_low_high_c4[2] = {
    _mm256_set_epi8(
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0,
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    ),
    _mm256_set_epi8(
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8,
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8
    )
  };
  const __m256i masks_low_high_c3[2] = {
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0,
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6,
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6
    )
  };
  const __m256i masks_lh_c3_c4[2] = {
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6,
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm256_set_epi8(
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8,
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    )
  };

  const __m128i masks_low128_c3_c4[2] = {
    _mm_set_epi8(
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm_set_epi8(
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    )
  };

  const auto mask_low = (num_channels == 3) ? masks_low_high_c3[0] : masks_low_high_c4[0];
  const auto mask_high = (num_channels == 3) ? masks_low_high_c3[1] : masks_low_high_c4[1];
  const auto mask_hl = (num_channels == 3) ? masks_lh_c3_c4[0] : masks_lh_c3_c4[1];
  const auto mask_low128 = (num_channels == 3) ? masks_low128_c3_c4[0] : masks_low128_c3_c4[1];

  // xsize = output width, xx = output x index
  // xmax = interpolation size, x = interpolation index (horizontal <-> x dimension)
  // xmin = input x start index corresponding to output x index (xx)

  const auto stride = num_channels * 1;  // num channels * sizeof(uint8)
  const auto zero = _mm_setzero_si128();

  for (const auto xx : c10::irange(xsize)) {
    __m128i sss;
    const auto xmin = xbounds[xx * 2 + 0];
    const auto xmax = xbounds[xx * 2 + 1];
    const auto k = &kk[xx * kmax];
    int64_t x = 0;

    if (xmax < 8) {
      sss = _mm_set1_epi32(1 << (coefs_precision - 1));
    } else {
      // Lower part will be added to higher, use only half of the error
      auto sss256 = _mm256_set1_epi32(1 << (coefs_precision - 2));

      // lineIn + stride * (x + xmin) + 32 <= lineIn + stride * (xmax + xmin)
      // --> x <= xmax - 32.0 / stride
      const auto b8_xmax = xmax - 32.0 / stride + 1;

      for (; x < b8_xmax; x += 8) {
        auto tmp = _mm_loadu_si128((__m128i*)&k[x]);
        auto ksource = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        auto source = _mm256_inserti128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *) (lineIn + stride * (x + 0 + xmin)))),
            _mm_loadu_si128((__m128i *) (lineIn + stride * (x + 4 + xmin))), 1);

        auto pix = _mm256_shuffle_epi8(source, mask_low);
        auto mmk = _mm256_shuffle_epi8(ksource, kmask_low);
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));

        pix = _mm256_shuffle_epi8(source, mask_high);
        mmk = _mm256_shuffle_epi8(ksource, kmask_high);
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
      }

      // lineIn + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
      // --> x <= xmax - 16.0 / stride
      const auto b4_xmax = xmax - 16.0 / stride + 1;

      for (; x < b4_xmax; x += 4) {
        auto tmp = _mm_loadu_si64(&k[x]);
        auto ksource = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        tmp = _mm_loadu_si128((__m128i*) (lineIn + stride * (x + xmin)));
        auto source = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        auto pix = _mm256_shuffle_epi8(source, mask_hl);
        auto mmk = _mm256_shuffle_epi8(ksource, kmask_hl);
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
      }

      sss = _mm_add_epi32(
          _mm256_extracti128_si256(sss256, 0),
          _mm256_extracti128_si256(sss256, 1));
    }

    // lineIn0 + stride * (x + xmin) + 8 <= lineIn0 + stride * (xmax + xmin)
    // --> x <= xmax - 8.0 / stride
    const auto b2_xmax = xmax - 8.0 / stride + 1;

    for (; x < b2_xmax; x += 2) {
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[x]);
      auto source = _mm_loadu_si64(lineIn + stride * (x + xmin));
      auto pix = _mm_shuffle_epi8(source, mask_low128);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    for (; x < xmax - 1; x++) {
      auto mmk = _mm_set1_epi32(k[x]);
      auto pix = mm_cvtepu8_epi32(lineIn + stride * (x + xmin));
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    if (x == xmax - 1) {
      // last element x = xmax - 1
      auto mmk = _mm_set1_epi32(k[x]);
      __m128i pix;
      auto p = lineIn + stride * (x + xmin);
      if (num_channels == 3) {
        unsigned char output[4];
        std::memcpy(output, p, 3);
        pix = mm_cvtepu8_epi32(output);
      } else {
        pix = mm_cvtepu8_epi32(p);
      }
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }
    sss = _mm_srai_epi32(sss, coefs_precision);
    sss = _mm_packs_epi32(sss, zero);
    sss = _mm_packus_epi16(sss, zero);
    if (num_channels == 3) {
      sss = _mm_shuffle_epi8(sss, mask_ch);
    }

    auto o = _mm_cvtsi128_si32(sss);
    if (num_channels == 3 && C10_UNLIKELY(stride * xx + 4 >= xsize * stride)) {
      std::memcpy(lineOut + stride * xx, (unsigned char *) &o, num_channels);
    } else {
      *(uint32_t *)(lineOut + stride * xx) = o;
    }
  }
}

void ImagingResampleVerticalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    int64_t ymin,
    int64_t ymax,
    const int16_t* k,
    unsigned int coefs_precision,
    int64_t num_channels) {

  // xsize = output width and input width, xx = output x index
  // ymax = interpolation size, y = interpolation index (vertical <-> y dimension)
  // ymin = input y start index

  const auto stride = num_channels * 1;  // num channels * sizeof(uint8)
  const int64_t data_size = xsize * stride;
  const int64_t data_stride = stride;
  constexpr auto vec_size = 256 / 8;

  const auto initial = _mm_set1_epi32(1 << (coefs_precision - 1));
  const auto initial_256 = _mm256_set1_epi32(1 << (coefs_precision - 1));
  const auto zero = _mm_setzero_si128();
  const auto zero_256 = _mm256_setzero_si256();

  const auto mask_ch = _mm_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 1, 0);

  int64_t i = 0;
  const auto b8_usable_vec_stride = (vec_size / data_stride) * data_stride;
  for (; i < data_size - vec_size; i += b8_usable_vec_stride) {
    auto sss0 = initial_256;
    auto sss1 = initial_256;
    auto sss2 = initial_256;
    auto sss3 = initial_256;
    int64_t y = 0;
    for (; y < ymax - 1; y += 2) {
      // Load two coefficients at once
      auto mmk = _mm256_set1_epi32(*(int32_t*)&k[y]);

      // Load 2 lines
      auto source1 =
          _mm256_loadu_si256((__m256i*)(lineIn + i + data_size * (y + ymin)));
      auto source2 =
          _mm256_loadu_si256((__m256i*)(lineIn + i + data_size * (y + 1 + ymin)));

      auto source = _mm256_unpacklo_epi8(source1, source2);
      auto pix = _mm256_unpacklo_epi8(source, zero_256);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, zero_256);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

      source = _mm256_unpackhi_epi8(source1, source2);
      pix = _mm256_unpacklo_epi8(source, zero_256);
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, zero_256);
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
    }
    for (; y < ymax; y += 1) {
      auto mmk = _mm256_set1_epi32(k[y]);

      auto source1 = _mm256_loadu_si256((__m256i*)(lineIn + i + data_size * (y + ymin)));

      auto source = _mm256_unpacklo_epi8(source1, zero_256);
      auto pix = _mm256_unpacklo_epi8(source, zero_256);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, zero_256);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

      source = _mm256_unpackhi_epi8(source1, zero_256);
      pix = _mm256_unpacklo_epi8(source, zero_256);
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
      pix = _mm256_unpackhi_epi8(source, zero_256);
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
    }
    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    sss2 = _mm256_srai_epi32(sss2, coefs_precision);
    sss3 = _mm256_srai_epi32(sss3, coefs_precision);

    sss0 = _mm256_packs_epi32(sss0, sss1);
    sss2 = _mm256_packs_epi32(sss2, sss3);
    sss0 = _mm256_packus_epi16(sss0, sss2);

    // Stores 32 bytes
    _mm256_storeu_si256((__m256i*)(lineOut + i), sss0);
  }

  const auto b2_usable_vec_stride = (8 / data_stride) * data_stride;
  // TODO: can we make b2_usable_vec_stride as (16 / data_stride) * data_stride ?
  for (; i < data_size - vec_size / 4; i += b2_usable_vec_stride) {
    auto sss0 = initial; // left row
    auto sss1 = initial; // right row
    int64_t y = 0;
    for (; y < ymax - 1; y += 2) {
      // Load two coefficients at once
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[y]);

      // Load 2 lines
      auto source1 = _mm_loadu_si64(lineIn + i + data_size * (y + ymin));
      auto source2 = _mm_loadu_si64(lineIn + i + data_size * (y + 1 + ymin));

      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
      pix = _mm_unpackhi_epi8(source, zero);
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
    }
    for (; y < ymax; y += 1) {
      auto mmk = _mm_set1_epi32(k[y]);

      auto source1 = _mm_loadu_si64(lineIn + i + data_size * (y + ymin));
      auto source = _mm_unpacklo_epi8(source1, zero);
      auto pix = _mm_unpacklo_epi8(source, zero);
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
      pix = _mm_unpackhi_epi8(source, zero);
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
    }
    sss0 = _mm_srai_epi32(sss0, coefs_precision);
    sss1 = _mm_srai_epi32(sss1, coefs_precision);

    sss0 = _mm_packs_epi32(sss0, sss1);
    sss0 = _mm_packus_epi16(sss0, sss0);

    _mm_storel_epi64((__m128i*)(lineOut + i), sss0);
  }

  const auto b1_usable_vec_stride = (4 / data_stride) * data_stride;
  for (; i < data_size - 4; i += b1_usable_vec_stride) {
    auto sss = initial;
    int64_t y = 0;
    for (; y < ymax - 1; y += 2) {
      // Load two coefficients at once
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[y]);

      // Load 2 lines
      auto source1 = _mm_cvtsi32_si128(*(int32_t*)(lineIn + i + data_size * (y + ymin)));
      auto source2 = _mm_cvtsi32_si128(*(int32_t*)(lineIn + i + data_size * (y + 1 + ymin)));

      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    for (; y < ymax; y++) {
      auto mmk = _mm_set1_epi32(k[y]);
      auto pix = mm_cvtepu8_epi32(lineIn + i + data_size * (y + ymin));
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }
    sss = _mm_srai_epi32(sss, coefs_precision);
    sss = _mm_packs_epi32(sss, zero);
    sss = _mm_packus_epi16(sss, zero);

    if (num_channels == 3) {
      sss = _mm_shuffle_epi8(sss, mask_ch);
    }

    auto o = _mm_cvtsi128_si32(sss);

    // Here we write 4 bytes to the output even if num_channels < 4, e.g o = {r,g,b,0} for num_channels=3
    // It is OK to write 4th byte (e.g. 0) as on the next step we will overwrite it with new data.
    // We also wont go out of bounds of lineOut memory allocation
    *(uint32_t *)(lineOut + i) = o;
  }

  for (; i < data_size; i += data_stride) {
    auto sss = initial;
    int64_t y = 0;
    for (; y < ymax - 1; y += 2) {
      // Load two coefficients at once
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[y]);

      // Load 2 lines
      auto source1 = _mm_cvtsi32_si128(*(int32_t*)(lineIn + i + data_size * (y + ymin)));
      auto source2 = _mm_cvtsi32_si128(*(int32_t*)(lineIn + i + data_size * (y + 1 + ymin)));

      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    for (; y < ymax; y++) {
      auto mmk = _mm_set1_epi32(k[y]);

      const uint8_t * p = lineIn + i + data_size * (y + ymin);
      __m128i pix;
      // TODO: Update condition to apply on the last pixel only
      if (num_channels == 3) {
        unsigned char output[4];
        std::memcpy(output, p, 3);
        pix = mm_cvtepu8_epi32(output);
      } else {
        pix = mm_cvtepu8_epi32(p);
      }
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    sss = _mm_srai_epi32(sss, coefs_precision);
    sss = _mm_packs_epi32(sss, zero);
    sss = _mm_packus_epi16(sss, zero);
    if (num_channels == 3) {
      sss = _mm_shuffle_epi8(sss, mask_ch);
    }

    auto o = _mm_cvtsi128_si32(sss);
    if (num_channels == 3 && C10_UNLIKELY(i + 4 >= data_size)) {
      std::memcpy(lineOut + i, (unsigned char *) &o, num_channels);
    } else {
      *(uint32_t *)(lineOut + i) = o;
    }
  }

}

}

using scale_t = std::vector<c10::optional<double>>;

int main(int argc, char** argv) {

    int n = 1000;
    int m = 7;
    int num_threads = 1;

    at::manual_seed(10);
    at::set_num_threads(num_threads);

    std::cout << "Torch config: " << at::show_config() << std::endl;
    std::cout << "Num threads: " << at::get_num_threads() << std::endl;

    auto size = 256;
    auto t_input = at::randint(0, 256, {1, 4, size, size}, at::CPU(at::kByte)).contiguous(at::MemoryFormat::ChannelsLast);

    std::cout << "\nInput tensor: " << t_input.sizes() << std::endl;
    std::cout << "Input is_contiguous memory_format torch.channels_last: "
              << (t_input.is_contiguous(at::MemoryFormat::ChannelsLast) ? "true" : "false") << std::endl;
    std::cout << "Input is_contiguous : "
              << (t_input.is_contiguous() ? "true" : "false") << std::endl;

    int64_t osizes[2] = {
        ((argc >= 2) ? std::atoi(argv[1]) : 224),
        ((argc >= 3) ? std::atoi(argv[2]) : 224)
    };
    at::IntArrayRef output_size(osizes);

    auto output = at::empty(
        {1, 4, osizes[0], osizes[1]}, t_input.options().memory_format(t_input.suggest_memory_format()));


    c10::optional<double> s1 = c10::nullopt;
    c10::optional<double> s2 = c10::nullopt;

    // warm-up
    for (int i=0; i<30000; i++) {
        test::upsample_avx_bilinear_uint8<scale_t, test::HelperInterpLinear>(
            t_input, output, false, {s1, s2}, /*antialias=*/true);
    }

    // // measure
    // std::cout << "\n- Bench _upsample_bilinear2d_aa (" << n << " rounds) - downsampling 256 -> (" << osizes[0] << ", " << osizes[1] << ")" << std::endl;
    // double avg_output = 0.0;

    // for (int j=0; j < m; j++) {
    //     auto start = std::chrono::steady_clock::now();
    //     for (int i=0; i<n; i++) {
    //         test::upsample_avx_bilinear_uint8<scale_t, test::HelperInterpLinear>(
    //             t_input, output, false, {s1, s2}, /*antialias=*/true);
    //     }
    //     auto end = std::chrono::steady_clock::now();
    //     std::chrono::duration<double> elapsed_seconds = end - start;
    //     avg_output += elapsed_seconds.count() / n;
    // }
    // avg_output /= m;

    // std::cout << "Elapsed time (us): " << avg_output * 1e6 << std::endl;

    return 0;
}