#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem> 


inline std::vector<char> LoadBytesFromFile(const std::string& filePath) {

    std::ifstream file(filePath, std::ios::in | std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return {}; 
    }

    size_t fileSize = std::filesystem::file_size(filePath);

    std::vector<char> buffer(fileSize);

  
    file.read(buffer.data(), fileSize);


    if (!file) {
        std::cerr << "Error: Only " << file.gcount() << " bytes were read." << std::endl;
        buffer.resize(file.gcount());
    }

    file.close();

    return buffer;
}