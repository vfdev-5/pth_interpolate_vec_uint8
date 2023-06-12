#include <iostream>


using scalar_t = double;

typedef scalar_t (*aa_filter_fn_t)(scalar_t);

template<typename scalar_t>
static inline scalar_t aa_filter_bicubic(scalar_t x) {
  // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
#define a -0.75
  if (x < 0.0) {
    x = -x;
  }
  if (x < 1.0) {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  }
  if (x < 2.0) {
    return (((x - 5) * x + 8) * x - 4) * a;
  }
  return 0.0;
#undef a
}


scalar_t _compute_weights_aa(
  const int64_t i, const int64_t input_size, const scalar_t scale, const scalar_t support,
  scalar_t* wt_ptr, const int64_t max_interp_size,
  int64_t& xmin, int64_t& xsize, bool antialias, double align_corners_delta
) {

  aa_filter_fn_t filter_fn = &aa_filter_bicubic;

  // align_corners_delta is 0.5 for uint8 and align_corners=true and antialias=false
  //                     is 0.0 otherwise
  scalar_t center = scale * (i + 0.5 - align_corners_delta);
  scalar_t total_w = 0.0;
  scalar_t invscale = (scale >= 1.0 && antialias) ? 1.0 / scale : 1.0;
  xmin = std::max(
      static_cast<int64_t>(center - support + 0.5 + align_corners_delta), static_cast<int64_t>(0));
  xsize = std::min(
      static_cast<int64_t>(center + support + 0.5 + align_corners_delta), input_size) - xmin;

  std::cout << "center: " << center << "\n";
  std::cout << "support: " << support << "\n";
  std::cout << "invscale: " << invscale << "\n";
  std::cout << "xmin: " << xmin << "\n";
  std::cout << "xsize: " << xsize << "\n";

  std::cout << "(center - support + 0.5 + align_corners_delta): " << center - support + 0.5 + align_corners_delta << "\n";
  std::cout << "(center + support + 0.5 + align_corners_delta): " << (center + support + 0.5 + align_corners_delta) << "\n";

  int64_t j = 0;
  std::cout << "unnorm w: ";
  for (; j < xsize; j++) {

    scalar_t w = filter_fn((j + xmin - center + 0.5 - align_corners_delta) * invscale);
    std::cout << w << " ";
    wt_ptr[j] = w;
    total_w += w;
  }
  std::cout << "\n";

  scalar_t wt_max = 0.0;
  if (total_w != 0.0) {
    std::cout << "w: ";
    for (j = 0; j < xsize; j++) {
      wt_ptr[j] /= total_w;
      std::cout << wt_ptr[j] << " ";
      wt_max = std::max(wt_max, wt_ptr[j]);
    }
    std::cout << "\n";
  }

  for (; j < max_interp_size; j++) {
    wt_ptr[j] = static_cast<scalar_t>(0.0);
  }
  return wt_max;
}


int main() {
  constexpr auto interp_size = 4;
  auto support = interp_size * 0.5;

  scalar_t weights[interp_size] = {0};

  double align_corners_delta = 0.0;

  int64_t xmin, xsize;
  _compute_weights_aa(
    1, 8, 2.0, support, weights, interp_size, xmin, xsize, false, align_corners_delta
  );

  return 0;
}