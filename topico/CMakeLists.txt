cmake_minimum_required(VERSION 3.0.2)
project(topico)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
)

set(SOURCES
src/a_O_P.cpp
src/a_O_V.cpp
src/ab_O_P.cpp
src/abc_O_A.cpp
src/abc_O_AP.cpp
src/abc_O_P.cpp
src/abc_O_V.cpp
src/abc_O_VP.cpp
src/abc_T_A.cpp
src/abc_T_P.cpp
src/abc_T_V.cpp
src/abcd_NO_P.cpp
src/abcd_O_P.cpp
src/abcde_NO_AP.cpp
src/abcde_NO_VP.cpp
src/abcde_O_AP.cpp
src/abcde_O_VP.cpp
src/abcde_T_A.cpp
src/abcde_T_P.cpp
src/abcde_T_V.cpp
src/abcdef_NO_VP.cpp
src/abcdef_O_VP.cpp
src/abcdef_T_P.cpp
src/abcdef_T_V.cpp
src/abcdefg_NO_AP.cpp
src/abcdefg_NO_AVP.cpp
src/abcdefg_NO_VP.cpp
src/abcdefg_NTV_AVP.cpp
src/abcdefg_O_AP.cpp
src/abcdefg_O_AVP.cpp
src/abcdefg_O_VP.cpp
src/abcdefg_TA_AVP.cpp
src/abcdefg_TV_AVP.cpp
src/abcdefg_T_A.cpp
src/abcdefg_T_AP.cpp
src/abcdefg_T_AV.cpp
src/abcdefg_T_P.cpp
src/abcdefg_T_V.cpp
src/abcdefg_T_VP.cpp
src/abcdeg_NO_AP.cpp
src/abcdeg_NO_AVP.cpp
src/abcdeg_NO_VP.cpp
src/abcdeg_NTV_AVP.cpp
src/abcdeg_O_AP.cpp
src/abcdeg_O_AVP.cpp
src/abcdeg_O_VP.cpp
src/abcdeg_TV_AVP.cpp
src/abcdeg_T_A.cpp
src/abcdeg_T_AP.cpp
src/abcdeg_T_AV.cpp
src/abcdeg_T_P.cpp
src/abcdeg_T_V.cpp
src/abcef_O_VP.cpp
src/abcef_T_P.cpp
src/abcef_T_V.cpp
src/abcefg_O_AP.cpp
src/abcefg_O_AVP.cpp
src/abcefg_TA_AVP.cpp
src/abcefg_T_A.cpp
src/abcefg_T_AP.cpp
src/abcefg_T_AV.cpp
src/abcefg_T_P.cpp
src/abcefg_T_V.cpp
src/abceg_O_AP.cpp
src/abceg_O_AVP.cpp
src/abceg_TA_AVP.cpp
src/abceg_T_A.cpp
src/abceg_T_AP.cpp
src/abceg_T_AV.cpp
src/abceg_T_P.cpp
src/abceg_T_V.cpp
src/abs.cpp
src/ac_O_A.cpp
src/ac_O_AP.cpp
src/ac_O_AV.cpp
src/ac_O_P.cpp
src/ac_O_V.cpp
src/ac_T_P.cpp
src/ac_T_V.cpp
src/acd_NO_P.cpp
src/acd_O_P.cpp
src/acde_NO_AP.cpp
src/acde_NO_VP.cpp
src/acde_O_AP.cpp
src/acde_O_VP.cpp
src/acde_T_A.cpp
src/acde_T_P.cpp
src/acde_T_V.cpp
src/acdef_NO_VP.cpp
src/acdef_O_VP.cpp
src/acdef_T_P.cpp
src/acdef_T_V.cpp
src/acdefg_NO_AP.cpp
src/acdefg_NO_AVP.cpp
src/acdefg_NO_VP.cpp
src/acdefg_NTA_AVP.cpp
src/acdefg_NTV_AVP.cpp
src/acdefg_O_AP.cpp
src/acdefg_O_AVP.cpp
src/acdefg_O_VP.cpp
src/acdefg_TV_AVP.cpp
src/acdefg_T_A.cpp
src/acdefg_T_AP.cpp
src/acdefg_T_AV.cpp
src/acdefg_T_P.cpp
src/acdefg_T_V.cpp
src/acdeg_NO_AP.cpp
src/acdeg_NO_AVP.cpp
src/acdeg_O_AP.cpp
src/acdeg_O_AVP.cpp
src/acdeg_TV_AVP.cpp
src/acdeg_T_A.cpp
src/acdeg_T_AP.cpp
src/acdeg_T_AV.cpp
src/acdeg_T_P.cpp
src/acdeg_T_V.cpp
src/acef_O_VP.cpp
src/acef_T_P.cpp
src/acef_T_V.cpp
src/acefg_O_AP.cpp
src/acefg_O_AVP.cpp
src/acefg_TA_AVP.cpp
src/acefg_T_A.cpp
src/acefg_T_AP.cpp
src/acefg_T_AV.cpp
src/acefg_T_P.cpp
src/acefg_T_V.cpp
src/aceg_O_AP.cpp
src/aceg_O_AVP.cpp
src/aceg_T_A.cpp
src/aceg_T_AP.cpp
src/aceg_T_AV.cpp
src/aceg_T_P.cpp
src/aceg_T_V.cpp
src/all.cpp
src/any1.cpp
src/anyNonFinite.cpp
src/check.cpp
src/check_feasibility.cpp
src/check_inputs.cpp
src/circshift.cpp
src/combineVectorElements.cpp
src/compensate_global.cpp
src/construct_setp_struct.cpp
src/correct_feasibility.cpp
src/cut_to_time.cpp
src/diff.cpp
src/eig.cpp
src/eigHermitianStandard.cpp
src/eml_int_forloop_overflow_check.cpp
src/evaluate_to_time.cpp
src/evolve_waypoints.cpp
src/find.cpp
src/i64ddiv.cpp
src/indexShapeCheck.cpp
src/minOrMax.cpp
src/mod.cpp
src/nnz.cpp
src/power.cpp
src/printint.cpp
src/repmat.cpp
src/rollout_t.cpp
src/roots.cpp
src/rotate_jerk.cpp
src/rtGetInf.cpp
src/rtGetNaN.cpp
src/rt_nonfinite.cpp
src/select_cases_O.cpp
src/select_cases_TA.cpp
src/select_cases_TV.cpp
src/select_conditions.cpp
src/select_type.cpp
src/simplify_setp.cpp
src/solve_O.cpp
src/solve_T.cpp
src/sort.cpp
src/sortIdx.cpp
src/sqrt.cpp
src/sub2ind.cpp
src/sum.cpp
src/synchronize_trajectory.cpp
src/topico_wrapper.cpp
src/topico_wrapper_data.cpp
src/topico_wrapper_initialize.cpp
src/topico_wrapper_rtwutil.cpp
src/topico_wrapper_terminate.cpp
src/useConstantDim.cpp
src/vAllOrAny.cpp
src/validateattributes.cpp
src/xdhseqr.cpp
src/xdlanv2.cpp
src/xnrm2.cpp
src/xrot.cpp
src/xzggbal.cpp
src/xzhgeqz.cpp
src/xzlanhs.cpp
src/xzlarf.cpp
src/xzlarfg.cpp
src/xzlartg.cpp)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

## Declare a C++ library
add_library(${PROJECT_NAME}
  ${SOURCES}
)


find_package(PythonInterp REQUIRED)
find_package(PythonLibs "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}" REQUIRED )
include(FetchContent)

FetchContent_Declare(
    pybind11-src
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.5.0
)

FetchContent_MakeAvailable(pybind11-src)

pybind11_add_module(pytopico SHARED ${LIB_SOURCE_FILES} ${LIB_HEADER_FILES} src/topico_python_bindings.cpp)
target_include_directories(pytopico
PRIVATE ${PYBIND11_INCLUDE_DIR}  # from project CMakeLists.txt
PRIVATE ${pybind11_INCLUDE_DIR}  # from pybind11Config
PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(pytopico PRIVATE
${PYTHON_LIBRARIES}
${CMAKE_THREAD_LIBS_INIT} topico)

install(TARGETS pytopico
				DESTINATION ${PYTHON_SITE_PACKAGES})
