#include <s8b.h>

S8bWord to_s8b(int count, int *vals) {
	S8bWord word;
	word.values = 0;
	word.selector = 0;
	int t = 0;
	//TODO: improve on this
	while(group_size[t] >= count && t < 16)
		t++;
	word.selector = t-1;
	unsigned long test = 0;
	for (int i = 0; i < count; i++) {
		test |= vals[count-i-1];
		if (i < count - 1)
			test <<= item_width[word.selector];
	}
	word.values = test;
		return word;
}