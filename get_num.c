#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
	if (argc != 3) {
		printf("usage ./get_num [num] p\n");
		return 1;
	}

	int num = atoi(argv[1]);
	int p = atoi(argv[2]);

	printf("using num: %d, p %d\n", num, p);

	int offset = 0;
	for (int i = 0; i < p; i++) {
		for (int j = i; j < p; j++) {
			if (offset == num) {
				printf("i: %d, j: %d\n", i+1, j+1);
			}
			offset++;
		}
	}
	printf("out of %d\n", offset + 1);
	return 0;
}
