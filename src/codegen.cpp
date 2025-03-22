#include "CodeGen.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

static constexpr int ORDER = 4;

std::vector<std::vector<std::pair<int, std::vector<int>>>> possibleSymmetries(int rank) {
	std::vector<std::vector<std::pair<int, std::vector<int>>>> result;
	if (rank > 1) {
		std::vector<int> indices(rank, 0);
		bool done = false;
		while (!done) {
			std::set<int> used;
			std::vector<std::pair<int, std::vector<int>>> thisResult;
			for (int i = 0; i < rank; i++) {
				int const thisIndex = indices[i];
				if (used.find(thisIndex) == used.end()) {
					std::pair<int, std::vector<int>> entry;
					entry.first = thisIndex >= 0;
					for (int j = 0; j < rank; j++) {
						if (indices[j] == thisIndex) {
							entry.second.push_back(j);
						}
					}
					if (entry.second.size() > 1) {
						thisResult.push_back(std::move(entry));
						used.insert(thisIndex);
					}
				}
			}
			result.push_back(std::move(thisResult));
			bool flag;
			do {
				flag = false;
				size_t r = 0;
				while (++indices[r] == rank) {
					indices[r++] = 0;
					if (r == indices.size()) {
						done = true;
						break;
					}
				}
				int last = -1;
				for (int r = 0; r < int(indices.size()); r++) {
					if (indices[r] > r) {
						flag = true;
						break;
					}
					bool rpt = false;
					for (int l = 0; l < r; l++) {
						if (indices[r] == indices[l]) {
							rpt = true;
							break;
						}
					}
					if (!rpt) {
						if (last >= 0) {
							if (indices[r] - indices[last] > 1) {
								flag = true;
								break;
							}
						}
						last = r;
					}
				}
			} while (flag);
		}
	}
	return result;
}

void includeFiles(CodeGen &code) {
	code.print("#include <array>");
	code.print("#include <cstddef>");
	code.newline();
}

void forwardDeclarations(CodeGen &code) {
	code.print("enum class symmetry_type : int {");
	code.indent();
	code.print("none, symmetric, antisymmetric");
	code.dedent();
	code.print("};");
	code.newline();
	code.print("template<char>");
	code.print("struct Index;");
	code.newline();
	code.print("template<symmetry_type, int...>");
	code.print("struct Symmetry;");
	code.newline();
	code.print("template<typename, size_t, size_t, typename...>");
	code.print("struct Tensor;");
	code.newline();
	code.print("template<typename, size_t, size_t, char...>");
	code.print("struct TensorExpression;");
	code.newline();
}

void TensorDeclaration(CodeGen &code, int rank) {
	std::string str;
	code.print("template<typename T, size_t D>");
	code.print("struct Tensor<T, D, %i> {", rank);
	code.indent();
	auto const accessOp = [&code, rank](bool constVersion) {
		std::string str;
		str = "T";
		str += constVersion ? " const" : "";
		str += "& operator()(";
		for (int r = 0; r < rank; r++) {
			str += "size_t ";
			str.push_back('i' + r);
			if (r + 1 < rank) {
				str += ", ";
			}
		}
		str += ")";
		str += constVersion ? " const;" : ";";
		code.print(str);
	};
	accessOp(false);
	accessOp(true);
	code.print("private:");
	str = "static constexpr size_t Size = ";
	if (rank == 0) {
		str += "1";

	} else {
		str += "D";
		for (int d = 1; d < rank; d++) {
			str += " * D";
		}
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
	auto const accessOp = [&code, rank](bool constVersion) {
		std::string str;
		code.print("template<typename T, size_t D>");
		str = "T";
		str += constVersion ? " const" : "";
		str += "& Tensor<T, D, " + std::to_string(rank) + ">::operator()(";
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
		str = "return V[";
		if (rank == 0) {
			str += "0";
		} else {
			for (int d = 1; d < rank; d++) {
				str += "D *";
				str += (d + 1 < rank) ? "(" : " ";
			}
			for (int d = rank - 1; d >= 0; d--) {
				str.push_back('i' + d);
				str += (d > 0) ? (std::string((d + 1 < rank) ? ")" : "") + " + ") : "";
			}
		}
		str += "];";
		code.print(str);
		code.dedent();
		code.print("}");
	};
	accessOp(false);
	code.newline();
	accessOp(true);
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
	for (int r = 0; r < rank; r++) {
		str += ", ";
		str.push_back('I' + r);
	}
	code.print("struct TensorExpression<T, D, %i%s> {", rank, str);
	code.indent();
	code.print("TensorExpression(T);");
	code.print("private:");
	code.print("T handle;");
	code.dedent();
	code.print("};");
	code.newline();
}

void expressionImplementation(CodeGen &code, int rank) {
	std::string str;
	str = "template<typename T, size_t D";
	str += rank ? " " : "";
	for (int r = 0; r < rank; r++) {
		str += ", ";
		str += "char ";
		str.push_back('I' + r);
	}
	str += ">";
	code.print(str);
	str = "";
	for (int r = 0; r < rank; r++) {
		str += ", ";
		str.push_back('I' + r);
	}
	code.print("TensorExpression<T, D, %i%s>::TensorExpression(T h) : handle(h) {", rank, str);
	code.indent();
	code.dedent();
	code.print("}");
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
	code.print("template<char C>");
	code.print("struct Index {");
	code.indent();
	code.print("static constexpr char value = C;");
	code.dedent();
	code.print("};");
	code.newline();
	code.print("template<symmetry_type S, int...Is>");
	code.print("struct Symmetry {");
	code.indent();
	code.print("static constexpr symmetry_type symmetryType = S;");
	code.print("static constexpr auto values = []() {");
	code.indent();
	code.print("std::array<int, sizeof...(Is)> values;");
	code.print("size_t index = 0;");
	code.print("((values[index++] = Is),...);");
	code.print("return values;");
	code.dedent();
	code.print("}();");
	code.dedent();
	code.print("};");
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
	for (int rank = 2; rank <= ORDER + 1; rank++) {
		auto const syms = possibleSymmetries(rank);
		printf("Rank = %i\n", rank);
		for (unsigned i = 0; i < syms.size(); i++) {
			auto const theseSyms = syms[i];
			for (unsigned j = 0; j < theseSyms.size(); j++) {
				printf("%c%i:", theseSyms[j].first ? '+' : '-', j);
				for (unsigned k = 0; k < theseSyms[j].second.size(); k++) {
					printf(" %i", theseSyms[j].second[k]);
				}
				if (j + 1 < theseSyms.size()) {
					printf(", ");
				}
			}
			if (theseSyms.size()) {
				printf("\n");
			}
		}
		printf("\n");
	}

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
}
