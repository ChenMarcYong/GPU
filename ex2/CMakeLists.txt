cmake_minimum_required(VERSION 3.16)
project(Histogram)

# Force libraries binaries to be with executable
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

enable_language(CUDA)

if(CMAKE_COMPILER_IS_GNUCC)
	set(
		HIST_COMPILATION_FLAGS
		-Wall
		-Wextra
	)
elseif(MSVC)
	set(HIST_COMPILATION_FLAGS
		/permissive
		/fp:fast
	)

	SET(
		HIST_COMPILATION_DEFINITION
		-D_CRT_SECURE_NO_WARNINGS
		-DNOMINMAX
	)
endif()

set(HIST_HEADERS
	"include/main.hpp"
	"include/generator.hpp"

	"include/references/referenceCPU.hpp"
	"include/references/referenceGPU.hpp"

	"include/utils/chronoCPU.hpp"
	"include/utils/chronoGPU.hpp"
	"include/utils/commonCUDA.hpp"
)

set(HIST_SOURCES
	"src/main.cpp"

	"src/references/referenceCPU.cpp"
	"src/references/referenceGPU.cu"

	"src/utils/chronoCPU.cpp"
	"src/utils/chronoGPU.cu"
)

# Force every file to by compiled with nvcc
set_source_files_properties(${HIST_SOURCES} PROPERTIES LANGUAGE CUDA)

add_executable(${PROJECT_NAME} ${HIST_SOURCES} ${HIST_HEADERS})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 11)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "70")
set_target_properties(${PROJECT_NAME} PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(${PROJECT_NAME} PRIVATE ${BCS_COMPILATION_FLAGS})
target_compile_definitions(${PROJECT_NAME} PRIVATE ${BCS_COMPILATION_DEFINITION})

target_include_directories(${PROJECT_NAME} PRIVATE SYSTEM "include/")