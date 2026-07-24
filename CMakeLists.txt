cmake_minimum_required(VERSION 3.18)
project(InferenceServer)

# 1. Configuración de C++ (Usamos 17 que es lo que pide la lib de tokens)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Agregar la librería de tokens (Asegurate que la carpeta sea external/tokenizers-cpp)
# Esto compila el código de Rust y C++ de la dependencia
set(TOKENIZER_PATH "${CMAKE_SOURCE_DIR}/external/tokenizers-cpp")
add_subdirectory(${TOKENIZER_PATH} tokenizers_build EXCLUDE_FROM_ALL)

# 3. Tu Ejecutable
# Agregamos tus archivos .cpp (main y el del servidor)
add_executable(server_exe 
    "src/main.cpp" 
    "src/serverHTTP.cpp"
    "src/server_queue.cpp"
    "src/ops.cpp"
    "src/layers.cpp"
    "external/LoadBytesFromFile.cpp" 
    "include/arenaAllocator.hpp" 
    "include/bi_gram_model.hpp"
    "src/bi_gram_model.cpp")

# 4. Incluir las carpetas de las cabeceras (.h)
target_include_directories(server_exe PRIVATE 
    ${CMAKE_SOURCE_DIR}                          # Para httplib.h si está en la raíz
    ${CMAKE_SOURCE_DIR}/include                  # Para headers del proyecto
    ${TOKENIZER_PATH}/include                    # Para tokenizers_cpp.h
)

# 5. Vincular las librerías
target_link_libraries(server_exe PRIVATE tokenizers_cpp)

# 6. Tests
enable_testing()

add_executable(arena_test
    "test/arena_test.cpp"
 "include/bi_gram_model.hpp" "src/bi_gram_model.cpp")

target_include_directories(arena_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

add_test(NAME ArenaAllocator COMMAND arena_test)

add_executable(ops_test
    "test/ops_test.cpp"
    "src/ops.cpp"
 "include/bi_gram_model.hpp" "src/bi_gram_model.cpp")

target_include_directories(ops_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

add_test(NAME OpsTest COMMAND ops_test)