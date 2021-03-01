#include "liblasso.h"

Queue *queue_new() {
	Queue *new_queue = malloc(sizeof(Queue));
	new_queue->length = 0;
	new_queue->first_item = NULL;
	new_queue->last_item = NULL;

	return new_queue;
}

int queue_is_empty(Queue *q) {
	if (q->length == 0)
		return TRUE;
	return FALSE;
}

void queue_push_tail(Queue *q, void *item) {
	struct Queue_Item *new_queue_item = malloc(sizeof(Queue_Item));
	new_queue_item->contents = item;
	new_queue_item->next = NULL;
	// if the queue is currently empty we set both first and last
	// item, rather than  last_item->next
	if (queue_is_empty(q)) {
		q->first_item = new_queue_item;
		q->last_item = new_queue_item;
	} else {
		q->last_item->next = new_queue_item;
		q->last_item = new_queue_item;
	}
	q->length++;
}

int queue_get_length(Queue *q) {
	return q->length;
}

void *queue_pop_head(Queue *q) {
	Queue_Item *first_item = q->first_item;
	if (first_item == NULL) {
		return NULL;
	}

	q->first_item = first_item->next;
	// if we pop'd the only item, don't keep it as last.
	if (NULL == q->first_item) {
		q->last_item = NULL;
	}
	q->length--;

	void *contents = first_item->contents;
	free(first_item);

	return contents;
}

/// Currently assumes that we want to free the contents
/// of everything in the queue as well
void queue_free(Queue *q) {
	Queue_Item *current_item = q->first_item;
	Queue_Item *next_item = current_item->next;

	// free the queue contents
	while (current_item != NULL) {
		free(current_item->contents);
		next_item = current_item->next;
		free(current_item);
		current_item = next_item;
	}
	// and the queue itself
	free(q);
}