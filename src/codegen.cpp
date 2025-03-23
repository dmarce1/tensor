#include "CodeGen.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <set>
#include <vector>

static constexpr int ORDER = 5;

template<size_t N>
static constexpr bool is_valid_permutation(const std::array<size_t, N> &arr) {
	constexpr std::array<size_t, N> ref = [] {
		std::array<size_t, N> r = { };
		for (size_t i = 0; i < N; ++i) {
			r[i] = i;
		}
		return r;
	}();
	return std::is_permutation(arr.begin(), arr.end(), ref.begin());
}

template<int S, size_t R, std::array<size_t, R> V>
struct Symmetry {
	static constexpr int sign = S;
	static constexpr std::array<size_t, R> values = V;
	static constexpr std::array<size_t, R> iota = [] {
		std::array<size_t, R> r = { };
		std::iota(r.begin(), r.end(), size_t(0));
		return r;
	}();
	Symmetry() {
		static_assert(std::is_permutation(V.begin(), V.end(), iota.begin()));
	}
};

template<size_t D, size_t R, size_t NS, size_t NA>
auto indexMap(std::array<std::array<size_t, R>, NS> sym, std::array<std::array<size_t, R>, NA> ant) {
	static constexpr size_t Size = std::pow(D, R);
	std::array<size_t, Size> map;
	std::array<int, Size> sgn;
	std::array<bool, Size> visited = { false };
	size_t s = Size;
	std::array<size_t, R> indices = { 0 };
	auto const I2i = [](std::array<size_t, R> indices) {
		size_t i = 0;
		for (size_t r = 0; r < R; r++) {
			i = D * i + indices[r];
		}
		return i;
	};
	auto const permuteIndices = [](std::array<size_t, R> indices, std::array<size_t, R> permutation) {
		std::array<size_t, R> result;
		for (size_t r = 0; r < R; r++) {
			result[r] = indices[permutation[r]];
		}
		return result;
	};
	size_t nextIndex = 0;
	while (s--) {
		size_t index = I2i(indices);
		if (!visited[index]) {
			static constexpr size_t symCount = 1 << (NS + NA);
			std::array<int, symCount> signs = { 0 };
			std::array<size_t, symCount> theseIndexes;
			bool zero = false;
			for (size_t bits = 0; bits < symCount; bits++) {
				int sgn = +1;
				size_t theseBits = bits;
				auto theseIndices = indices;
				for (size_t k = 0; k < NS; k++) {
					if (theseBits & 1) {
						theseIndices = permuteIndices(theseIndices, sym[k]);
					}
					theseBits >>= 1;
				}
				for (size_t k = 0; k < NA; k++) {
					if (theseBits & 1) {
						theseIndices = permuteIndices(theseIndices, ant[k]);
						if (theseIndices == indices) {
							sgn = 0;
							zero = true;
						} else {
							sgn = -sgn;
						}
					}
					theseBits >>= 1;
				}
				signs[bits] = sgn;
				size_t const thisIndex = I2i(theseIndices);
				theseIndexes[bits] = thisIndex;
				visited[thisIndex] = true;
			}
			for (size_t bits = 0; bits < symCount; bits++) {
				if (!zero) {
					map[theseIndexes[bits]] = nextIndex;
					sgn[theseIndexes[bits]] = signs[bits];
				} else {
					map[theseIndexes[bits]] = std::numeric_limits<size_t>::max();
					sgn[theseIndexes[bits]] = 0;
				}
			}
			if (!zero) {
				nextIndex++;
			}
		}
		if (s) {
			std::reverse(indices.begin(), indices.end());
			int r = 0;
			while (++indices[r] == D) {
				indices[r++] = 0;
			}
			std::reverse(indices.begin(), indices.end());
		}
	}
	return std::make_pair(map, sgn);
}

void indexMapDeclaration(CodeGen &code) {
	const char *headerString = "template<size_t, size_t R, size_t N, size_t M>\n"
			"static constexpr auto indexMap(std::array<std::array<size_t, R>, N>, std::array<std::array<size_t, R>, M>);\n";
	code.stringToFile(headerString);

}
void indexMap2Header(CodeGen &code) {
	const char *codeString = "template<size_t D, size_t R, size_t N, size_t M>\n"
			"static constexpr auto indexMap(std::array<std::array<size_t, R>, N> sym, std::array<std::array<size_t, R>, M> ant) {\n"
			"    static constexpr size_t Size = std::pow(D, R);\n"
			"    std::array<size_t, Size> map;\n"
			"    std::array<int, Size> sgn;\n"
			"    std::array<bool, Size> visited = { false };\n"
			"    size_t s = Size;\n"
			"    std::array<size_t, R> indices = { 0 };\n"
			"    auto const I2i = [](std::array<size_t, R> indices) {\n"
			"        size_t i = 0;\n"
			"        for (size_t r = 0; r < R; r++) {\n"
			"            i = D * i + indices[r];\n"
			"        }\n"
			"        return i;\n"
			"    };\n"
			"    auto const permuteIndices = [](std::array<size_t, R> indices, std::array<size_t, R> permutation) {\n"
			"        std::array<size_t, R> result;\n"
			"        for (size_t r = 0; r < R; r++) {\n"
			"            result[r] = indices[permutation[r]];\n"
			"        }\n"
			"        return result;\n"
			"    };\n"
			"    size_t nextIndex = 0;\n"
			"    while (s--) {\n"
			"        size_t index = I2i(indices);\n"
			"        if (!visited[index]) {\n"
			"            static constexpr size_t symCount = 1 << (N + M);\n"
			"            std::array<int, symCount> signs = { 0 };\n"
			"            std::array<size_t, symCount> theseIndexes;\n"
			"            bool zero = false;\n"
			"            for (size_t bits = 0; bits < symCount; bits++) {\n"
			"                int sgn = +1;\n"
			"                size_t theseBits = bits;\n"
			"                auto theseIndices = indices;\n"
			"                for (size_t k = 0; k < N; k++) {\n"
			"                    if (theseBits & 1) {\n"
			"                        theseIndices = permuteIndices(theseIndices, sym[k]);\n"
			"                    }\n"
			"                    theseBits >>= 1;\n"
			"                }\n"
			"                for (size_t k = 0; k < M; k++) {\n"
			"                    if (theseBits & 1) {\n"
			"                        theseIndices = permuteIndices(theseIndices, ant[k]);\n"
			"                        if (theseIndices == indices) {\n"
			"                            sgn = 0;\n"
			"                            zero = true;\n"
			"                        } else {\n"
			"                            sgn = -sgn;\n"
			"                        }\n"
			"                    }\n"
			"                    theseBits >>= 1;\n"
			"                }\n"
			"                signs[bits] = sgn;\n"
			"                size_t const thisIndex = I2i(theseIndices);\n"
			"                theseIndexes[bits] = thisIndex;\n"
			"                visited[thisIndex] = true;\n"
			"            }\n"
			"            for (size_t bits = 0; bits < symCount; bits++) {\n"
			"                if (!zero) {\n"
			"                    map[theseIndexes[bits]] = nextIndex;\n"
			"                    sgn[theseIndexes[bits]] = signs[bits];\n"
			"                } else {\n"
			"                    map[theseIndexes[bits]] = std::numeric_limits<size_t>::max();\n"
			"                    sgn[theseIndexes[bits]] = 0;\n"
			"                }\n"
			"            }\n"
			"            if (!zero) {\n"
			"                nextIndex++;\n"
			"            }\n"
			"        }\n"
			"        if (s) {\n"
			"            std::reverse(indices.begin(), indices.end());\n"
			"            int r = 0;\n"
			"            while (++indices[r] == D) {\n"
			"                indices[r++] = 0;\n"
			"            }\n"
			"            std::reverse(indices.begin(), indices.end());\n"
			"        }\n"
			"    }\n"
			"    return std::make_pair(map, sgn);\n"
			"};\n";
	code.stringToFile(codeString);
}

std::string tensorTypeString(int rank) {
	std::string str = "Tensor<T, D, " + std::to_string(rank);
	str += ">";
	return str;
}

void TensorDeclaration(CodeGen &code, int rank) {
	std::string str;
	auto const typeString = tensorTypeString(rank);
	code.print("template<typename T, size_t D>");
	code.print("struct %s {", typeString);
	code.indent();
	auto const accessOp = [&code, rank](bool constVersion) {
		std::string str;
		str = "constexpr T";
		str += constVersion ? " const" : "";
		str += "& operator()(";
		for (int r = 0; r < rank; r++) {
			str += "size_t";
			str += (r + 1 < rank) ? ", " : "";
		}
		str += ")";
		str += constVersion ? " const;" : ";";
		code.print(str);
		size_t count = 1 << rank;
		for (size_t bits = 1; bits < count; bits++) {
			std::string str;
			str = "template<";
			bool first = true;
			std::string charString1;
			std::string charString2;
			for (int r = 0; r < rank; r++) {
				if ((bits >> r) & 1) {
					if (!first) {
						charString1 += ", ";
						charString2 += ", ";
					}
					charString1 += "char ";
					charString1.push_back('I' + r);
					charString2.push_back('I' + r);
					first = false;
				}
			}
			str += charString1;
			str += "> ";
			code.print(str);
			str = "constexpr auto ";
			str += "operator()(";
			for (int r = 0; r < rank; r++) {
				if ((bits >> r) & 1) {
					str += "Index<";
					str.push_back('I' + r);
					str += ">";
					str += (r + 1 < rank) ? ", " : "";
				} else {
					str += "size_t";
					str += (r + 1 < rank) ? ", " : "";
				}
			}
			str += ")";
			str += constVersion ? " const;" : ";";
			code.print(str);
		}
	};
	accessOp(false);
	accessOp(true);
	code.print("private:");
	str = "static constexpr int computeIndex(";
	for (int r = 0; r < rank; r++) {
		str += "size_t ";
		str.push_back('i' + r);
		if (r + 1 < rank) {
			str += ", ";
		}
	}
	str += ");";
	code.print(str);
	int genRank = rank;
	str = "static constexpr size_t Size = ";
	if (genRank) {
		if (genRank > 1) {
			str += "(";
		}
		str += "D";
		for (int d = 1; d < genRank; d++) {
			str += " * D";
		}
		if (genRank > 1) {
			str += ")";
		}
	} else {
		str += "1";
	}
	str += ";";
	code.print(str);
	code.print("std::array<T, Size> V;");
	code.dedent();
	code.print("};");
	code.newline();
}

void TensorImplementation(CodeGen &code, int rank) {
	std::string str;
	int genRank = rank;
	auto const typeString = tensorTypeString(rank);
	auto const accessOp = [&code, rank, typeString](bool constVersion) {
		std::string str;
		code.print("template<typename T, size_t D>");
		str = "constexpr T";
		str += constVersion ? " const" : "";
		str += "& " + typeString + "::operator()(";
		for (int r = 0; r < rank; r++) {
			str += "size_t ";
			str.push_back('i' + r);
			if (r + 1 < rank) {
				str += ", ";
			}
		}
		str += ")";
		str += constVersion ? " const" : "";
		str += " {";
		code.print(str);
		code.indent();
		str = "int const index = computeIndex(";
		for (int r = 0; r < rank; r++) {
			str.push_back('i' + r);
			if (r + 1 < rank) {
				str += ", ";
			}
		}
		str += ");";
		code.print(str);
		code.print("return V[index];");
		code.dedent();
		code.print("}");
		size_t count = 1 << rank;
		for (size_t bits = 1; bits < count; bits++) {
			std::string str;
			code.newline();
			code.print("template<typename T, size_t D>");
			str = "template<";
			bool first = true;
			std::string charString1;
			std::string charString2;
			for (int r = 0; r < rank; r++) {
				if ((bits >> r) & 1) {
					if (!first) {
						charString1 += ", ";
						charString2 += ", ";
					}
					charString1 += "char ";
					charString1.push_back('I' + r);
					charString2.push_back('I' + r);
					first = false;
				}
			}
			str += charString1;
			str += "> ";
			code.print(str);
			str = "constexpr auto ";
			str += typeString + "::operator()(";
			for (int r = 0; r < rank; r++) {
				if ((bits >> r) & 1) {
					str += "Index<";
					str.push_back('I' + r);
					str += ">";
					str += (r + 1 < rank) ? ", " : "";
				} else {
					str += "size_t ";
					str.push_back('i' + r);
					str += (r + 1 < rank) ? ", " : "";
				}
			}
			str += ")";
			str += constVersion ? " const {" : " {";
			code.print(str);
			code.indent();

			str = "auto f = [this";
			for (int r = 0; r < rank; r++) {
				if (!((bits >> r) & 1)) {
					str += ", ";
					str.push_back('i' + r);
				}
			}
			str += "](";
			first = true;
			for (int r = 0; r < rank; r++) {
				if ((bits >> r) & 1) {
					str += !first ? ", " : "";
					str += "size_t ";
					str.push_back('i' + r);
					first = false;
				}
			}
			str += ") {";
			code.print(str);
			code.indent();
			str = "";
			first = true;
			for (int r = 0; r < rank; r++) {
				str += !first ? ", " : "";
				str.push_back('i' + r);
				first = false;
			}
			code.print("return this->operator()(%s);", str);
			code.dedent();
			code.print("};");
			code.print("return TensorExpression<decltype(f), D, %i, %s>(std::move(f));", rank, charString2);
			code.dedent();
			code.print("}");
		}
	};
	accessOp(false);
	code.newline();
	accessOp(true);
	code.newline();
	code.print("template<typename T, size_t D>");
	str = "constexpr int " + typeString + "::computeIndex(";
	for (int r = 0; r < rank; r++) {
		str += "size_t ";
		str.push_back('i' + r);
		if (r + 1 < rank) {
			str += ", ";
		}
	}
	str += ") {";
	code.indent();
	code.print(str);
	str = "size_t index = ";
	if (genRank == 0) {
		str += "0;";
	} else {
		for (int d = 1; d < genRank; d++) {
			str += "D";
			str += (d + 1 < genRank) ? " * (" : " * ";
		}
		for (int d = 0; d < genRank; d++) {
			str += (d != 0) ? "+ " : "";
			str.push_back('i' + d);
			str += ((d > 0) && (d != genRank - 1)) ? ") " : " ";
		}
		str.pop_back();
	}
	str += ";";
	code.print(str);
	code.print("return index;");
	code.dedent();
	code.print("}");
	code.newline();
}

void expressionDeclaration(CodeGen &code, int rank) {
	std::string str;
	str = "template<typename T, size_t D";
	for (int r = 0; r < rank; r++) {
		str += ", char ";
		str.push_back('I' + r);
	}
	str += ">";
	code.print(str);
	str = "";
	std::vector<char> charString;
	for (int r = 0; r < rank; r++) {
		str += ", ";
		str.push_back('I' + r);
		charString.push_back('I' + r);
	}
	code.print("struct TensorExpression<T, D, %i%s> {", rank, str);
	code.indent();
	code.print("TensorExpression(T);");
	do {
		str = "constexpr auto operator=(TensorExpression<T, D, " + std::to_string(rank);
		for (int r = 0; r < rank; r++) {
			str += ", ";
			str.push_back(charString[r]);
		}
		str += "> const&);";
		code.print(str);

	} while (std::next_permutation(charString.begin(), charString.end()));
	code.print("private:");
	code.print("T handle;");
	code.dedent();
	code.print("};");
	code.newline();
}

void expressionImplementation(CodeGen &code, int rank) {
	std::string str;
	str = "template<typename T, size_t D";
	for (int r = 0; r < rank; r++) {
		str += ", ";
		str += "char ";
		str.push_back('I' + r);
	}
	str += ">";
	std::string const tempStr = str;
	code.print(str);
	str = "";
	std::vector<char> charString;
	for (int r = 0; r < rank; r++) {
		str += ", ";
		str.push_back('I' + r);
		charString.push_back('I' + r);
	}
	std::string const typeString = "TensorExpression<T, D, " + std::to_string(rank) + str + ">";
	code.print("%s::TensorExpression(T h) : handle(h) {", typeString);
	code.indent();
	code.dedent();
	code.print("}");
	code.newline();
	do {
		code.print("%s", tempStr);
		str = "constexpr auto " + typeString + "::operator=(TensorExpression<T, D, " + std::to_string(rank);
		for (int r = 0; r < rank; r++) {
			str += ", ";
			str.push_back(charString[r]);
		}
		str += "> const& other) {";
		code.print(str);
		code.indent();
		for (int r = 0; r < rank; r++) {
			char const c = 'i' + r;
			code.print("for (size_t %c = 0; %c < D; %c++) {", c, c, c);
			code.indent();
		}
		str = "handle(";
		for (int r = 0; r < rank; r++) {
			str += r ? ", " : "";
			char const c = 'i' + r;
			str.push_back(c);
		}
		str += ") = other(";
		for (int r = 0; r < rank; r++) {
			str += r ? ", " : "";
			str.push_back(charString[r] + 'a' - 'A');
		}
		str += ");";
		code.print(str);
		for (int r = 0; r < rank; r++) {
			code.dedent();
			code.print("}");
		}
		code.dedent();
		code.print("}");
		code.newline();
	} while (std::next_permutation(charString.begin(), charString.end()));
	code.newline();
}

void forwardDeclarations(CodeGen &code) {
	code.newline();
	code.print("template<char>");
	code.print("struct Index;");
	indexMapDeclaration(code);
	code.print("template<typename, size_t, size_t>");
	code.print("struct Tensor;");
	code.newline();
	code.print("template<typename, size_t, size_t, char...>");
	code.print("struct TensorExpression;");
	code.stringToFile(""
			"template<int S, size_t R, std::array<size_t, R> V>\n"
			"struct Symmetry;"
			"\n"
			"");
}

void helpers(CodeGen &code) {
	code.newline();
	code.print("template<char C>");
	code.print("struct Index {");
	code.indent();
	code.print("static constexpr char value = C;");
	code.dedent();
	code.print("};");
	indexMap2Header(code);
	code.stringToFile(""
			"template<int S, size_t R, std::array<size_t, R> V>\n"
			"struct Symmetry {\n"
			"\tstatic constexpr int sign = S;\n"
			"\tstatic constexpr std::array<size_t, R> values = V;\n"
			"\tstatic constexpr std::array<size_t, R> iota = [] {\n"
			"\t\tstd::array<size_t, R> r = { };\n"
			"\t\tstd::iota(r.begin(), r.end(), size_t(0));\n"
			"\t\treturn r;\n"
			"\t}();\n"
			"\tSymmetry() {\n"
			"\t\tstatic_assert(std::is_permutation(V.begin(), V.end(), iota.begin()));\n"
			"\t}\n"
			"};\n"
			"");
}

void includeFiles(CodeGen &code) {
	code.print("#include <algorithm>");
	code.print("#include <array>");
	code.print("#include <cmath>");
	code.print("#include <cstddef>");
	code.print("#include <numeric>");
	code.print("#include <stdexcept>");
	code.print("#include <tuple>");
	code.print("#include <utility>");
	code.newline();
}

std::string generate() {
	CodeGen code;
	code.print("#pragma once");
	code.newline();
	includeFiles(code);
	code.newline();
	code.print("namespace Tensors {");
	code.newline();
	code.sectionComment("Forward Declarations");
	forwardDeclarations(code);
	code.newline();
	code.sectionComment("Helper Classes");
	helpers(code);
	code.newline();
	code.sectionComment("Tensor Declarations");
	for (int r = 0; r <= ORDER; r++) {
		TensorDeclaration(code, r);
	}
	code.sectionComment("Expression Declarations");
	for (int r = 0; r <= ORDER; r++) {
		expressionDeclaration(code, r);
	}
	code.newline();
	code.sectionComment("Tensor Implementations");
	for (int r = 0; r <= ORDER; r++) {
		TensorImplementation(code, r);
	}
	code.sectionComment("Expression Implementations");
	for (int r = 0; r <= ORDER; r++) {
		expressionImplementation(code, r);
	}
	code.newline();
	code.print("}");
	code.newline();
	return code.get();
}

int main(int argc, char *argv[]) {
//	for (int rank = 2; rank <= ORDER + 1; rank++) {
//		auto const syms = possibleSymmetries(rank);
//		printf("Rank = %i\n", rank);
//		for (unsigned i = 0; i < syms.size(); i++) {
//			auto const theseSyms = syms[i];
//			for (unsigned j = 0; j < theseSyms.size(); j++) {
//				printf("%c%i:", theseSyms[j].first ? '+' : '-', j);
//				for (unsigned k = 0; k < theseSyms[j].second.size(); k++) {
//					printf(" %i", theseSyms[j].second[k]);
//				}
//				if (j + 1 < theseSyms.size()) {
//					printf(", ");
//				}
//			}
//			if (theseSyms.size()) {
//				printf("\n");
//			}
//		}
//		printf("\n");
//	}
// =
//	static constexpr size_t D = 3;
//	static constexpr size_t R = 3;
//	static constexpr size_t NS = 0;
//	static constexpr size_t NA = 3;
//	static constexpr std::array<std::array<size_t, R>, NS> sym = { { } };
//	static constexpr std::array<std::array<size_t, R>, NA> ant = { { { 1, 0, 2 }, { 2, 1, 0 }, { 0, 2, 1 } } };
////	static constexpr size_t D = 4;
////	static constexpr size_t R = 4;
////	static constexpr size_t NS = 1;
////	static constexpr size_t NA = 2;
////	static constexpr std::array<std::array<size_t, R>, NS> sym = { { { 2, 3, 0, 1 } } };
////	static constexpr std::array<std::array<size_t, R>, NA> ant = { { { 1, 0, 2, 3 }, { 0, 1, 3, 2 } } };
//	auto imap = indexMap<D, R, NS, NA>(sym, ant);
//	for (size_t i = 0; i < imap.first.size(); i++) {
//		size_t k = i;
//		for (int r = 0; r < R; r++) {
//			printf("%2i ", k % D);
//			k /= D;
//		}
//		if (imap.first[i] != std::numeric_limits<size_t>::max()) {
//			printf("| %li | %s %li\n", i, imap.second[i] > 0 ? "+" : "-", imap.first[i]);
//		} else {
//			printf("| %li | %s *\n", i, imap.second[i] > 0 ? "+" : "-");
//		}
//	}

	int rc = -1;
	try {
		if (argc >= 2) {
			std::ofstream file(argv[1]);
			if (file.is_open()) {
				file << generate();
				std::cout << "Code generation successful.\n";
				rc = 0;
			}
		}
		if (rc != 0) {
			throw std::runtime_error("Failed to open the file.\n");
		}
	} catch (const std::exception &exception) {
		std::cerr << exception.what() << std::endl;
	}
	return rc;
	return 0;
}
