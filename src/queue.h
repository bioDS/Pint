typedef struct Queue_Item {
	void *contents;
	void *next;
} Queue_Item;

typedef struct Queue {
	Queue_Item *first_item;
	Queue_Item *last_item;
	int length;
} Queue;