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
    main.cpp 
    serverHTTP.cpp
    server_queue.cpp
    "external/LoadBytesFromFile.cpp")

# 4. Incluir las carpetas de las cabeceras (.h)
target_include_directories(server_exe PRIVATE 
    ${CMAKE_SOURCE_DIR}                          # Para httplib.h si está en la raíz
    ${TOKENIZER_PATH}/include                    # Para tokenizers_cpp.h
)

# 5. Vincular las librerías
target_link_libraries(server_exe PRIVATE tokenizers_cpp)