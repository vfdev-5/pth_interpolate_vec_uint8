cmake_minimum_required(VERSION 3.18)
project(check)

# As ~PyTorch
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx -mfma -mavx2 -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -DNDEBUG -O3 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -mavx -mfma -mavx2 -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -DNDEBUG -O3 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow")

find_package(Torch REQUIRED)

# add_executable(scratch scratch.cpp)
# target_link_libraries(scratch PUBLIC ${TORCH_LIBRARIES})
# set_property(TARGET scratch PROPERTY CXX_STANDARD 17)

# add_executable(check_ti_channels_last check_ti_channels_last.cpp)
# target_link_libraries(check_ti_channels_last PUBLIC ${TORCH_LIBRARIES})
# set_property(TARGET check_ti_channels_last PROPERTY CXX_STANDARD 17)

add_executable(check_decomp_upsampling check_decomp_upsampling.cpp)
target_link_libraries(check_decomp_upsampling PUBLIC ${TORCH_LIBRARIES})
set_property(TARGET check_decomp_upsampling PROPERTY CXX_STANDARD 17)

# add_executable(check_ti_channels_first check_ti_channels_first.cpp)
# target_link_libraries(check_ti_channels_first PUBLIC ${TORCH_LIBRARIES})
# set_property(TARGET check_ti_channels_first PROPERTY CXX_STANDARD 17)

# add_executable(profile_vect_interp profile_vect_interp.cpp)
# target_link_libraries(profile_vect_interp PUBLIC ${TORCH_LIBRARIES})
# set_property(TARGET profile_vect_interp PROPERTY CXX_STANDARD 17)

# add_executable(check_mem_format_3d_input check_mem_format_3d_input.cpp)
# target_link_libraries(check_mem_format_3d_input PUBLIC ${TORCH_LIBRARIES})
# set_property(TARGET check_mem_format_3d_input PROPERTY CXX_STANDARD 17)

# add_executable(unpack_rgb_with_copy unpack_rgb_with_copy.cpp)
# target_link_libraries(unpack_rgb_with_copy PUBLIC ${TORCH_LIBRARIES})
# set_property(TARGET unpack_rgb_with_copy PROPERTY CXX_STANDARD 17)

# add_executable(check_cpp_api_bicubic_issue check_cpp_api_bicubic_issue.cpp)
# target_link_libraries(check_cpp_api_bicubic_issue PUBLIC ${TORCH_LIBRARIES})
# set_property(TARGET check_cpp_api_bicubic_issue PROPERTY CXX_STANDARD 17)
