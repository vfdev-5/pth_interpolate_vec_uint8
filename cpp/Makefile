# CC=g++ -std=c++14 -mavx -O3
# CC=g++ -std=c++14 -mavx -mavx2 -O3 -mfma
CC=g++ -std=c++17 -O3 -mavx -mfma -mavx2 -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -O3 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow

check_resize_avx_rgb: check_resize_avx_rgb.cpp
	$(CC) check_resize_avx_rgb.cpp -o check_resize_avx_rgb


check_resize_avx_rgba: check_resize_avx_rgba.cpp
	$(CC) check_resize_avx_rgba.cpp -o check_resize_avx_rgba


check_resize_avx_rgba_CF: check_resize_avx_rgba_CF.cpp
	$(CC) check_resize_avx_rgba_CF.cpp -o check_resize_avx_rgba_CF


play_avx: play_avx.cpp
	$(CC) play_avx.cpp -o play_avx


check_mem_boundaries_with_asan: check_mem_boundaries_with_asan.cpp
	clang++ check_mem_boundaries_with_asan.cpp -o check_mem_boundaries_with_asan -O2 -g -fsanitize=address -fno-omit-frame-pointer
