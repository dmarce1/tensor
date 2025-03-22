#include <cstdio>
#include "codegen.hpp"

int main(int argc, char* argv[]) {
	printf( "%s\n", argv[1]);
	FILE* fp = fopen(argv[1], "wt");
	fclose(fp);
	return 0;
}
