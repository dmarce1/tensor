#pragma once

#include <cstdio>
#include <cstring>
#include <string>

template<typename T>
struct ArgConverter {
	static T convert(const T &arg) {
		return arg;
	}
};

template<>
struct ArgConverter<std::string> {
	static const char* convert(const std::string &arg) {
		return arg.c_str();
	}
};

template<typename ... Args>
int printf_wrapper(const std::string &format, Args &&... args) {
	return std::printf(format.c_str());
}

struct CodeGen {
	static constexpr char const *const tabString = "   ";
	CodeGen() {
		tabs = 0;
		nl = false;
	}
	template<typename ... Args>
	void print(std::string const &fmt, Args &&...args) {
		std::string line;
		char *buffer;
		asprintf(&buffer, fmt.c_str(), ArgConverter<typename std::decay<Args>::type>::convert(std::forward<Args>(args))...);

		for (int n = 0; n < std::strlen(buffer); n++) {
			code.push_back(buffer[n]);
			if (buffer[n] == '\n') {
				for (int t = 0; t < tabs; t++) {
					code += tabString;
				}
			}
		}
		free(buffer);
	}
	void indent() {
		tabs++;
	}
	void dedent() {
		tabs--;
	}
	std::string get() const {
		return code;
	}
private:
	int tabs;
	bool nl;
	std::string code;
};

