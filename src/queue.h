typedef struct Queue_Item {
  void *contents;
  void *next;
} Queue_Item;

typedef struct Queue {
  Queue_Item *first_item;
  Queue_Item *last_item;
  int length;
} Queue;

Queue *queue_new();
int queue_is_empty(Queue *q);
void queue_push_tail(Queue *q, void *item);
int queue_get_length(Queue *q);
void *queue_pop_head(Queue *q);
void queue_free(Queue *q);