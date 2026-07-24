#pragma once
#include <vector>
#include <stdexcept>


class Arena {
	std::vector<std::byte> buffer_;
	size_t offset_ = 0;

public:
	explicit Arena(size_t storage_size) : buffer_(storage_size) {}
	void* alloc(size_t bytes) {
		if (offset_ + bytes > buffer_.size()) { // we give less than the 100% of our storage
			throw std::out_of_range("Arena allocation exceeds capacity");
		}
		void* ptr = buffer_.data() + offset_;
		offset_ += bytes;
		return ptr;
	}
	void reset() { offset_ = 0; } // to delete memory
};