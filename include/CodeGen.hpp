#pragma once

#include <cstdio>
#include <cstring>
#include <stdexcept>
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

struct CodeGen {
	static constexpr char const *const indentString = "   ";
	CodeGen() {
		indentCount = 0;
	}
	std::string getIndention() const {
		std::string str;
		for (int n = 0; n < indentCount; n++) {
			str += indentString;
		}
		return str;
	}
	void stringToFile(std::string const str) {
		generatedCode += "\n" + str + "\n";
	}
	void print(std::string const &line) {
		if (line[0] != '#' && line.back() != ':') {
			generatedCode += getIndention();
		}
		generatedCode += line;
		generatedCode += "\n";
	}
	template<typename ... Args>
	void print(std::string const &fmt, Args &&...args) {
		char *buffer;
		int const rc = asprintf(&buffer, fmt.c_str(), ArgConverter<typename std::decay<Args>::type>::convert(std::forward<Args>(args))...);
		if (rc < 0) {
			throw std::runtime_error("Error in asprintf call.\n");
		}
		std::string const line(buffer);
		print(line);
		free(buffer);
	}
	void indent() {
		indentCount++;
	}
	void dedent() {
		indentCount--;
	}
	void newline(int N = 1) {
		for (int n = 0; n < N; n++) {
			generatedCode += getIndention();
			generatedCode += "\n";
		}
	}
	std::string get() const {
		return generatedCode;
	}
	void sectionComment(std::string const &comment) {
		generatedCode += "/******************************************************************************/\n/* " + comment;
		for (unsigned i = 0; i < 75 - comment.size(); i++) {
			generatedCode += " ";
		}
		generatedCode += "*/\n";
		generatedCode += "/******************************************************************************/\n\n";
	}
private:
	int indentCount;
	std::string generatedCode;
};

