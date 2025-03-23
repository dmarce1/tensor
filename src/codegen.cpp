#include "CodeGen.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>
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
			size_t const count = 1 << thisResult.size();
			for (size_t i = 0; i < count; i++) {
				auto copy = thisResult;
				for (size_t j = 0; j < thisResult.size(); j++) {
					copy[j].first = ((i >> j) & 1);
				}
				result.push_back(std::move(copy));
			}
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

std::string tensorTypeString(int rank, std::vector<std::pair<int, std::vector<int>>> const &symmetry) {
	std::string str = "Tensor<T, D, " + std::to_string(rank);
	for (auto symGroup : symmetry) {
		str += symGroup.first ? ", Symmetry<" : ", Antisymmetry<";
		for (size_t j = 0; j < symGroup.second.size(); j++) {
			str += std::to_string(symGroup.second[j]);
			str += (j + 1 < symGroup.second.size()) ? ", " : "";
		}
		str += ">";
	}
	str += ">";
	return str;
}

void TensorDeclaration(CodeGen &code, int rank) {
	std::string str;
	auto symmetries = possibleSymmetries(rank);
	if (symmetries.size() == 0) {
		symmetries.resize(1);
	}
	for (auto const &symmetry : symmetries) {
		auto const typeString = tensorTypeString(rank, symmetry);
		code.print("template<typename T, size_t D>");
		code.print("struct %s {", typeString);
		code.indent();
		auto const accessOp = [&code, rank](bool constVersion) {
			std::string str;
			str = "constexpr T";
			str += constVersion ? " const" : "";
			str += "& operator[](";
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
//				str += "constexpr TensorExpression<Tensor const&, D, " + std::to_string(rank) + ", " + charString2 + ">";
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
		std::vector<std::string> symStrings;
		for (auto symGroup : symmetry) {
			auto const &sym = symGroup.second;
			str = "(D";
			int fac = 1;
			genRank--;
			for (size_t l = 1; l < sym.size(); l++) {
				str += " * ";
				if (!symGroup.first) {
					str += "std::max";
				}
				str += "(D " + std::string(symGroup.first ? "+" : "-") + " " + std::to_string(l);
				if (!symGroup.first) {
					str += ", size_t(0)";
				}
				str += +")";
				fac *= l + 1;
				genRank--;
			}
			if (fac > 1) {
				str += " / " + std::to_string(fac);
			}
			str += ")";
			symStrings.push_back(std::move(str));
		}
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
		} else if (symmetry.size() == 0) {
			str += "1";
		}
		bool first = true;
		for (auto const &s : symStrings) {
			str += (genRank || !first) ? " * " : "";
			str += s;
			first = false;
		}
		str += ";";
		code.print(str);
		code.print("std::array<T, Size> V;");
		code.dedent();
		code.print("};");
		code.newline();
	}
}

void TensorImplementation(CodeGen &code, int rank) {
	std::string str;
	auto symmetries = possibleSymmetries(rank);
	if (symmetries.size() == 0) {
		symmetries.resize(1);
	}
	for (auto const &symmetry : symmetries) {
		std::unordered_map<char, int> sMap;
		auto const typeString = tensorTypeString(rank, symmetry);
		for (int i = 0; i < rank; i++) {
			sMap['i' + i] = 0;
		}
		int genRank = rank;
		bool hasAntisymmetry = false;
		for (auto symGroup : symmetry) {
			hasAntisymmetry = hasAntisymmetry || !symGroup.first;
			for (size_t j = 0; j < symGroup.second.size(); j++) {
				sMap['i' + symGroup.second[j]] = 1;
				genRank--;
			}
		}
		auto const accessOp = [&code, rank, typeString, hasAntisymmetry](bool constVersion) {
			std::string str;
			code.print("template<typename T, size_t D>");
			str = "constexpr T";
			str += constVersion ? " const" : "";
			str += "& " + typeString + "::operator[](";
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
			if (hasAntisymmetry) {
				code.print("if( index > 0 ) {");
				code.indent();
				code.print("return V[index - 1];");
				code.dedent();
				code.print("} else if( index < 0 ) {");
				code.indent();
				code.print("return -V[-(index + 1)];");
				code.dedent();
				code.print("} else /*if( index == 0 )*/ {");
				code.indent();
				if (constVersion) {
					code.print("return T(0);");
				} else {
					code.print("throw std::runtime_error(\"Exception in T %s::operator[](...).\\n\");", typeString);
				}
				code.dedent();
				code.print("}");
			} else {
				code.print("return V[index];");
			}
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
//				str += "constexpr TensorExpression<" + typeString + " const&, D, " + std::to_string(rank) + ", " + charString2 + ">";
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
				str = "return TensorExpression<" + typeString + " const&, D, " + std::to_string(rank) + ", " + charString2 + ">([this";
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
				code.print("});");
				code.dedent();
				code.print("}");
			}
		};
		accessOp(false);
		code.newline();
		accessOp(true);
		code.newline();
		std::string str;
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
		code.print(str);
		code.indent();
		int si = 0;
		bool flag = true;
		for (auto symGroup : symmetry) {
			if (symGroup.second.size()) {
				auto const &sym = symGroup.second;
				if (si == 0) {
					str = "static constexpr size_t size" + std::to_string(si) + " = D";
				} else {
					str = "static constexpr size_t size" + std::to_string(si) + " = size" + std::to_string(si - 1) + " * D";
				}
				int fac = 1;
				for (size_t l = 1; l < sym.size(); l++) {
					fac *= l + 1;
					str += " * ";
					if (!symGroup.first) {
						str += "std::max";
					}
					str += "(D " + std::string(symGroup.first ? "+" : "-") + " " + std::to_string(l);
					if (!symGroup.first) {
						str += ", size_t(0)";
					}
					str += +")";
				}
				if (fac > 1) {
					str += " / " + std::to_string(fac);
				}
				str += ";";
				code.print(str);
				str = "";
				si++;
				flag = false;
			}
		}
		if (hasAntisymmetry) {
			code.print("int sign = 1;");
		}
		str = "size_t index = ";
		if (genRank == 0) {
			str += "0;";
		} else {
			auto it = sMap.begin();
			for (int d = 1; d < genRank; d++) {
				str += "D * ";
				str += (d + 1 < genRank) ? "(" : " ";
			}
			for (int d = genRank - 1; d >= 0; d--) {
				while (it->second != 0) {
					it++;
				}
				str.push_back(it->first);
				str += (d > 0) ? (std::string((d + 1 < genRank) ? ")" : "") + " + ") : ";";
				it++;
			}
		}
		code.print(str);
		for (auto symGroup : symmetry) {
			if (symGroup.second.size()) {
				auto const &sym = symGroup.second;
				for (size_t k = 0; k < sym.size(); k++) {
					for (size_t l = k + 1; l < sym.size(); l++) {
						char const c1 = 'i' + sym[k];
						char const c2 = 'i' + sym[l];
						code.print("if(%c < %c) {", c1, c2);
						code.indent();
						code.print("std::swap(%c, %c);", c1, c2);
						if (hasAntisymmetry) {
							code.print("sign = -sign;");
						}
						code.dedent();
						if (!symGroup.first) {
							code.print("} else if( %c == %c ) {", c1, c2);
							code.indent();
							code.print("sign = 0;");
							code.dedent();
						}
						code.print("}");
					}
				}
			}
		}
		si = 0;
		for (auto symGroup : symmetry) {
			if (symGroup.second.size()) {
				auto const &sym = symGroup.second;
				code.print("index *= size%i;", si);
				for (size_t k = 0; k < sym.size(); k++) {
					std::string ch(1, 'i' + sym[k]);
					str = "index += " + ch;
					int fac = 1;
					for (size_t l = 1; l < sym.size() - k; l++) {
						fac *= l + 1;
						str += " * (" + ch + std::string(symGroup.first ? " + " : " - ") + std::to_string(l) + ")";
					}
					if (fac > 1) {
						str += " / " + std::to_string(fac);
					}
					str += ";";
					code.print(str);
				}
				si++;
			}
		}
		if (hasAntisymmetry) {
			code.print("index++;");
			code.print("if( sign == +1 ) {");
			code.indent();
			code.print("return index;");
			code.dedent();
			code.print("} else if( sign == -1 ) {");
			code.indent();
			code.print("return -index;");
			code.dedent();
			code.print("} else /*if( sign == 0 )*/ {");
			code.indent();
			code.print("return 0;");
			code.dedent();
			code.print("}");
		} else {
			code.print("return index;");
		}
		code.dedent();
		code.print("}");
		code.newline();
	}
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

void forwardDeclarations(CodeGen &code) {
	code.newline();
	code.print("template<size_t, size_t, size_t...>");
	code.print("struct Antisymmetry;");
	code.newline();
	code.print("template<char>");
	code.print("struct Index;");
	code.newline();
	code.print("template<size_t, size_t, size_t...>");
	code.print("struct Symmetry;");
	code.newline();
	code.print("template<typename, size_t, size_t, typename...>");
	code.print("struct Tensor;");
	code.newline();
	code.print("template<typename, size_t, size_t, char...>");
	code.print("struct TensorExpression;");
	code.newline();
}

void helpers(CodeGen &code) {
	code.print("template<size_t I0, size_t I1, size_t...Is>");
	code.print("struct Antisymmetry {");
	code.indent();
	code.print("static constexpr std::array values = {I0, I1, Is...};");
	code.dedent();
	code.print("};");
	code.newline();
	code.print("template<char C>");
	code.print("struct Index {");
	code.indent();
	code.print("static constexpr char value = C;");
	code.dedent();
	code.print("};");
	code.newline();
	code.print("template<size_t I0, size_t I1, size_t...Is>");
	code.print("struct Symmetry {");
	code.indent();
	code.print("static constexpr std::array values = {I0, I1, Is...};");
	code.dedent();
	code.print("};");
	code.newline();
}

void includeFiles(CodeGen &code) {
	code.print("#include <algorithm>");
	code.print("#include <array>");
	code.print("#include <cstddef>");
	code.print("#include <stdexcept>");
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
//
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
