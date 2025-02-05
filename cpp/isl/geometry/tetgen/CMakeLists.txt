file(
	GLOB
	ISL_TETGEN_HEADERS
	*.h
	)

file(
	GLOB
	ISL_TETGEN_SOURCES
	*.cpp
	)
  
add_library(
	isl_tetgen
	STATIC
 	${ISL_TETGEN_SOURCES}
	${ISL_TETGEN_HEADERS}
	)

set_target_properties(
	isl_tetgen
	PROPERTIES
	PUBLIC_HEADER "${ISL_TETGEN_HEADERS}"
	)

target_compile_definitions(
	isl_tetgen
	PRIVATE
	$<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_WARNINGS>
	$<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_DEPRECATE>
	$<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
	$<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_DEPRECATE>
	$<$<CXX_COMPILER_ID:MSVC>:_SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS>
	$<$<CXX_COMPILER_ID:GNU>:_USE_MATH_DEFINES>
	$<$<CXX_COMPILER_ID:GNU>:GL_SILENCE_DEPRECATION>
	$<$<CXX_COMPILER_ID:AppleClang>:_USE_MATH_DEFINES>
	$<$<CXX_COMPILER_ID:AppleClang>:GL_SILENCE_DEPRECATION>
	)

target_include_directories(
	isl_tetgen
	PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/>
	$<INSTALL_INTERFACE:include>
	PRIVATE
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/isl/geometry/tetgen/>
	)

target_link_libraries(
	isl_tetgen
	PUBLIC
	Eigen3::Eigen
	tetgen
	)
	
install(
	TARGETS isl_tetgen
	ARCHIVE DESTINATION lib
	PUBLIC_HEADER DESTINATION include
	)

set_target_properties(
	isl_tetgen
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/python"
	RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/python"
	)
