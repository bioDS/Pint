#include "liblasso.h"
#include <omp.h>
#include <glib-2.0/glib.h>
//#include <ncurses.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_permutation.h>
#include <errno.h>
#include <sys/time.h>
#ifdef USE_R
#include <R.h>
#else
#define Rprintf printf
#endif

#define NumCores 4
#define NumSets  1024
#define LIMIT_OVERLAP

const static int NORMALISE_Y = 0;
int skipped_updates = 0;
int total_updates = 0;
int skipped_updates_entries = 0;
int total_updates_entries = 0;
static int zero_updates = 0;
static int zero_updates_entries = 0;

static int VERBOSE = 1;
static int haschanged = 1;
static int *colsum;
static double *col_ysum;
//static double max_rowsum = 0;

#define NUM_MAX_ROWSUMS 50
static double max_rowsums[NUM_MAX_ROWSUMS];
static double max_cumulative_rowsums[NUM_MAX_ROWSUMS];

static gsl_permutation *global_permutation;
static gsl_permutation *global_permutation_inverse;

int min(int a, int b) {
	if (a < b)
		return a;
	return b;
}

static int N;

//TODO: try using dancing links
//TODO: stop after |set| = NumSets?
//		- maybe after NumSets*10 (or something) has been allowd through this row
//TODO: split into GList[NumSets] (or similar) (by rows w/ no overlap?)
//		OR: pre-allocate GList contents?
//TODO: rather than linked lists, arrays of structs with offsets might compress better?
// GQueue?

Column_Set copy_column_set(Column_Set from_set) {
	ColEntry *from = from_set.cols;
	int max_size = from_set.size;
	GSList *tlist = NULL;
	int count = 0;
	for (int i = 0; i < max_size;) {
		if (from[i].nextEntry > 0) {
			count++;
			tlist = g_slist_prepend(tlist, (void*)(long)from[i].value);
			i = from[i].nextEntry;
		} else if (from[i].nextEntry < 0) {
			i = -from[i].nextEntry;
		} else {
			fprintf(stderr, "nextEntry was 0\n");
		}
	}

	tlist = g_slist_reverse(tlist);
	int length = g_slist_length(tlist);
	Column_Set new_set;
	new_set.size = length;
	new_set.cols = malloc(length*sizeof(ColEntry));
	count = 0;
	for (GSList *temp = tlist; temp != NULL; temp = temp->next) {
		new_set.cols[count].value = (int)(long)temp->data;
		new_set.cols[count].nextEntry = count + 1;
		count++;
	}

	return new_set;
}

//TODO: don't bother merging sets with too many elements?
int can_merge(Mergeset *all_sets, int i1, int i2, double frac_overlap_allowed) {
	//if ((all_sets[i1].size > 20) || (all_sets[i2].size > 20))
	////if ((all_sets[i1].size > N/20 && all_sets[i1].ncols == 1) || (all_sets[i2].size > N/20 && all_sets[i2].ncols == 1))
	//	return FALSE;
	if (all_sets[i1].size == 0 || all_sets[i2].size == 0)
		return TRUE;

	int max_size = 0;
	if (all_sets[i1].size > all_sets[i2].size)
		max_size = all_sets[i1].size;
	else
		max_size = all_sets[i2].size;

	int allowable_overlap = (int)(frac_overlap_allowed*max_size);
	int ti1 = 0, ti2 = 0, used_overlap = 0;
	while (ti1 < all_sets[i1].size && ti2 < all_sets[i2].size) {
		while (ti1 < all_sets[i1].size && all_sets[i1].entries[ti1] < all_sets[i2].entries[ti2]) {
			ti1++;
		}
		if (ti1 >= all_sets[i1].size)
			return TRUE;
		while (ti2 < all_sets[i2].size && all_sets[i2].entries[ti2] < all_sets[i1].entries[ti1]) {
			ti2++;
		}
		if (ti2 >= all_sets[i2].size)
			return TRUE;
		if (all_sets[i1].entries[ti1] == all_sets[i2].entries[ti2])
			if (used_overlap++ > allowable_overlap) {
				return FALSE;
			} else {
				ti1++; ti2++;
			}
	}
	return TRUE;
}

// merges set i2 into set i1
void merge_sets(Mergeset *all_sets, int i1, int i2) {
	int indices[all_sets[i1].size + all_sets[i2].size];
	int used_rows = 0;

	int ti1 = 0, ti2 = 0;
	while (ti1 < all_sets[i1].size && ti2 < all_sets[i2].size) {
		while (ti1 < all_sets[i1].size && all_sets[i1].entries[ti1] < all_sets[i2].entries[ti2]) {
			indices[ti1 + ti2] = all_sets[i1].entries[ti1];
			ti1++;
			used_rows++;
		}
		if (ti1 >= all_sets[i1].size)
			break;
		while (ti2 < all_sets[i2].size && all_sets[i2].entries[ti2] < all_sets[i1].entries[ti1]) {
			indices[ti1 + ti2] = all_sets[i2].entries[ti2];
			ti2++;
			used_rows++;
		}
		if (ti2 >= all_sets[i2].size)
			break;
		if (all_sets[i1].entries[ti1] == all_sets[i2].entries[ti2]) {
			//fprintf(stderr, "attempted to merge unmergeable sets\n");
			//fprintf(stderr, "set %d entry %d = %d, set %d entry %d = %d\n", i1, ti1, all_sets[i1].entries[ti1], i2, ti2, all_sets[i2].entries[ti2]);
			//return;
			ti1++;
			ti2++;
		///(*(int*)0)++;
			//TODO: don't leave this in here
		}
	}
	// we've done the overlap, now read in the rest
	while (ti1 < all_sets[i1].size) {
		indices[ti1++ + ti2] = all_sets[i1].entries[ti1];
		used_rows++;
	}
	while (ti2 < all_sets[i2].size) {
		indices[ti1 + ti2++] = all_sets[i2].entries[ti2];
		used_rows++;
	}

	int *actual_indices = malloc(used_rows*sizeof(int));
	memcpy(actual_indices, indices, used_rows*sizeof(int));
	free(all_sets[i1].entries);
	all_sets[i1].entries = actual_indices;
	all_sets[i1].size = used_rows;

	int *new_cols = malloc((all_sets[i1].ncols + all_sets[i2].ncols)*sizeof(int));
	memcpy(new_cols, all_sets[i1].cols, all_sets[i1].ncols*sizeof(int));
	memcpy(&new_cols[all_sets[i1].ncols], all_sets[i2].cols, all_sets[i2].ncols*sizeof(int));
	free(all_sets[i1].cols);
	free(all_sets[i2].cols);
	all_sets[i1].cols = new_cols;
	all_sets[i2].cols = NULL;
	all_sets[i1].ncols = all_sets[i1].ncols + all_sets[i2].ncols;
}

Mergeset *remove_invalid_sets(Mergeset *all_sets, int *valid_mergesets, int actual_p_int, int new_mergeset_count, int *actual_set_sizes) {
	Mergeset *all_sets_new = malloc(new_mergeset_count *sizeof(Mergeset));

	int old_counter = 0;
	for (int i = 0; i < new_mergeset_count; i++) {
		while(valid_mergesets[old_counter] == FALSE)
			old_counter++;
		all_sets_new[i] = all_sets[old_counter];
		memcpy(&all_sets_new[i], &all_sets[old_counter], sizeof(Mergeset));
		//actual_set_sizes[i] = actual_set_sizes[old_counter];
		old_counter++;
	}
	free(all_sets);
	return all_sets_new;
}

// check the first n elements of set_bins_of_size[small] against the (offset +) first n of set_bins_of_size[large]
// (modulo their respective sizes the indices of those that can be merged are placed in sets_to_merge.
// TODO: allow different offsets for small and large (to allow small == large)
int compare_n(Mergeset *all_sets, int *valid_mergesets, int **set_bins_of_size, int *num_bins_of_size, int *sets_to_merge, int small, int large, int n, int small_offset, int large_offset, double frac_overlap_allowed) {
	int num_bins_to_merge = 0;
	int small_set_no, large_set_no;
	small_offset = small_offset%num_bins_of_size[small];
	large_offset = large_offset%num_bins_of_size[large];
	if (small == large && small_offset == large_offset) {
		fprintf(stderr, "comparing a set with itself using the same offset won't work\n");
		return 0;
	}

	#pragma omp parallel for shared(set_bins_of_size, valid_mergesets) private(small_set_no, large_set_no) reduction(+:num_bins_to_merge)
	for (int i = 0; i < n; i++) {
		small_set_no = set_bins_of_size[small][(i + small_offset) %num_bins_of_size[small]];
		large_set_no = set_bins_of_size[large][(i + large_offset) %num_bins_of_size[large]];
		//TODO: we shouldn't need the valid_mergesets check here if everything is working
		if (valid_mergesets[small_set_no] == TRUE && valid_mergesets[large_set_no] == TRUE && can_merge(all_sets, small_set_no, large_set_no, frac_overlap_allowed) == TRUE) {
			sets_to_merge[i] = 1;
			num_bins_to_merge++;
		}
		else
			sets_to_merge[i] = 0;
	}
	return num_bins_to_merge;
}


int max(int a, int b) {
	if (a > b)
		return a;
	return b;
}

// if we wrap around either the large or small set, the new version will preserve the order, but not the starting position.
// TODO: allow merging with the same set size (very important)
// TODO: allow merging with the max set size (not so important)
void merge_n(Mergeset *all_sets, int **set_bins_of_size, int *num_bins_of_size, int *valid_mergesets, int *sets_to_merge, int small, int large, int n, int small_offset, int large_offset, int num_bins_to_merge) {
	large_offset = large_offset%num_bins_of_size[large];
	small_offset = small_offset%num_bins_of_size[small];
	int new_small_offset = 0, new_large_offset = 0;
	int end_pos_small, end_pos_large;
	int smallest_offset, largest_offset, end_pos_smallest, end_pos_largest;
	int small_beginning = small_offset;

	if (num_bins_to_merge == 0)
		return;

	int max_set_size, small_set_no, large_set_no;
	if (large+small > NumSets)
		max_set_size = NumSets;
	else
		max_set_size = large+small;

	if (max_set_size == large) {
		// this isn't implemented yet, segfault
		(*(int*)0)++;
	}

	int *new_small_bin = malloc((num_bins_of_size[small]        - num_bins_to_merge)*sizeof(int));
	int *new_large_bin = NULL;
	int *new_xl_bin = NULL;
	new_xl_bin    = malloc((num_bins_of_size[max_set_size] + num_bins_to_merge)*sizeof(int));
	new_large_bin = malloc((num_bins_of_size[large]        - num_bins_to_merge)*sizeof(int));

	// copy old contents of sets that will not be overwritten

	// first, copy the old xl bin, and the old small/large sets up to the offest
	memcpy(new_xl_bin, set_bins_of_size[max_set_size], num_bins_of_size[max_set_size]*sizeof(int));
	if (small == large && large_offset < small_offset)
		smallest_offset = large_offset;
	if (small == large) {
		smallest_offset = min(small_offset, large_offset);
		largest_offset = max(small_offset, large_offset);
		end_pos_smallest = (smallest_offset+n)%num_bins_of_size[small];
		end_pos_largest = (largest_offset+n)%num_bins_of_size[small];
		small_beginning = smallest_offset;
		if (end_pos_smallest > largest_offset) {
			fprintf(stderr, "merging overlapping regions (%d-%d) (%d-%d) in the same set (%d), segfaulting instead\n",
					smallest_offset, end_pos_smallest, largest_offset, end_pos_largest, small);
			(*(int*)0)++;
		}
		if (end_pos_largest < largest_offset) {
			// the second group has wrapped around
			if (end_pos_largest > smallest_offset) {
				fprintf(stderr, "merging overlapping regions (%d-%d) (%d-%d) in the same set (%d), segfaulting instead\n",
						smallest_offset, end_pos_smallest, largest_offset, end_pos_largest, small);
				(*(int*)0)++;
			}
			// but not too far
			memcpy(new_small_bin, &set_bins_of_size[small][end_pos_largest], (small_offset - end_pos_largest)*sizeof(int));
			small_beginning = smallest_offset - end_pos_largest;
		} else {
			// or the second group did not wrap around, and we copy from 0
			memcpy(new_small_bin, &set_bins_of_size[small][0], (small_offset)*sizeof(int));
		}
	} else {
		// if we wrapped around the set, copy middle, otherwise copy the beginning and end separately
		end_pos_small = (small_offset+n)%num_bins_of_size[small];
		end_pos_large = (large_offset+n)%num_bins_of_size[large];

		new_small_offset = 0;
		new_large_offset = 0;
		if (end_pos_small == small_offset)
			new_small_offset = -small_offset;
		if (end_pos_large == large_offset)
			new_large_offset = -large_offset;

		if (end_pos_small > small_offset)
			memcpy(new_small_bin, &set_bins_of_size[small][0], small_offset*sizeof(int));
			// copy the rest later to preserve the order of elements being merged.
		else if (end_pos_small < small_offset) {
			memcpy(new_small_bin, &set_bins_of_size[small][end_pos_small], (small_offset-end_pos_small)*sizeof(int));
			new_small_offset = -end_pos_small;
		}

		if (end_pos_large > large_offset)
			memcpy(new_large_bin, &set_bins_of_size[large][0], large_offset*sizeof(int));
			// copy the rest later to preserve the order of elements being merged.
		else if (end_pos_large < large_offset) {
			memcpy(new_large_bin, &set_bins_of_size[large][end_pos_large], (large_offset-end_pos_large)*sizeof(int));
			new_large_offset = -end_pos_large;
		}
	}

	int merged_count = 0, unmerged_count = 0;
	for (int i = 0; i < n; i++) {
		small_set_no = set_bins_of_size[small][(i+small_offset)%num_bins_of_size[small]];
		large_set_no = set_bins_of_size[large][(i+large_offset)%num_bins_of_size[large]];
		if (sets_to_merge[i] == 1) {
			//printf("merging %d,%d\n", small_set_no, large_set_no);
			merge_sets(all_sets, small_set_no, large_set_no);
			new_xl_bin[num_bins_of_size[max_set_size]+merged_count] = small_set_no;
			merged_count++;
			valid_mergesets[large_set_no] = FALSE;
		} else {
			if (small != large) {
				//printf("writing to %d out of %d\n", small_offset+unmerged_count + new_small_offset,
				//									num_bins_of_size[small] - num_bins_to_merge);
				if ((small_offset+unmerged_count) + new_small_offset > num_bins_of_size[small] - num_bins_to_merge)
					(*(int*)0)++;
				new_small_bin[(small_offset+unmerged_count) + new_small_offset] = small_set_no;
				if ((large_offset+unmerged_count) + new_large_offset > num_bins_of_size[large] - num_bins_to_merge)
					(*(int*)0)++;
				new_large_bin[(large_offset+unmerged_count) + new_large_offset] = large_set_no;
			} else { //the two are the same set, don't duplicate entries
				new_small_bin[(small_beginning+unmerged_count++) + new_small_offset] = small_set_no;
				new_small_bin[(small_beginning+unmerged_count) + new_small_offset] = large_set_no;
			}
			unmerged_count++;
		}
	}

	// copy the rest of the initial list if there is any
	if (small == large) {
		// in this case we also have to copy the middle
		memcpy(&new_small_bin[smallest_offset+unmerged_count], &set_bins_of_size[small][end_pos_smallest], (largest_offset - end_pos_smallest)*sizeof(int));
		if (end_pos_largest > largest_offset) {
			// If the end had wrapped around, we would have already copied the beginning.
			// We only need to copy the end.
			memcpy(&new_small_bin[smallest_offset+unmerged_count + largest_offset - end_pos_smallest], &set_bins_of_size[small][end_pos_largest], (num_bins_of_size[small] - end_pos_largest)*sizeof(int));
		}
	} else {
		if (end_pos_small > small_offset)
			memcpy(&new_small_bin[small_offset+unmerged_count], &set_bins_of_size[small][small_offset+n], (num_bins_of_size[small] - n - small_offset)*sizeof(int));
		if (end_pos_large > large_offset)
			memcpy(&new_large_bin[large_offset+unmerged_count], &set_bins_of_size[large][large_offset+n], (num_bins_of_size[large] - n - large_offset)*sizeof(int));
	}

	free(set_bins_of_size[small]);
	if (small != large) {
		free(set_bins_of_size[large]);
		set_bins_of_size[large] = new_large_bin;
	}
	free(set_bins_of_size[max_set_size]);

	set_bins_of_size[small] = new_small_bin;
	set_bins_of_size[max_set_size] = new_xl_bin;
	num_bins_of_size[small] -= num_bins_to_merge;
	num_bins_of_size[large] -= num_bins_to_merge;
	num_bins_of_size[max_set_size] += num_bins_to_merge;
}

//TODO: don't allocate so many arrays on the stack?
Beta_Sets merge_find_beta_sets(XMatrix_sparse x2col, int actual_p_int, int n, double frac_overlap_allowed) {
	Rprintf("allowing overlap of %.2f%% finding beta sets\n", (frac_overlap_allowed*100));
	Mergeset *all_sets = malloc(actual_p_int*sizeof(Mergeset));
	int *valid_mergesets = malloc(actual_p_int*sizeof(int));
	//int actual_set_sizes[actual_p_int+1];
	int *valid_mergeset_indices = malloc(actual_p_int*sizeof(int));
	int mergeset_count = actual_p_int;
	int new_mergeset_count;
	Beta_Sets return_sets;


	long total_col_nz = 0;
	for (int i = 0; i < actual_p_int; i++)
		valid_mergesets[i] = TRUE;

	for (int i = 0; i < actual_p_int; i++) {
		total_col_nz += x2col.col_nz[i];
		all_sets[i].size = x2col.col_nz[i];
		all_sets[i].cols = malloc(sizeof(int));
		all_sets[i].cols[0] = i;
		all_sets[i].ncols = 1;
		all_sets[i].entries = malloc(all_sets[i].size*sizeof(int));
		memcpy(all_sets[i].entries, x2col.col_nz_indices[i], all_sets[i].size*sizeof(int));
		//actual_set_sizes[i] = 1;
	}
	//printw("mean col_nz: %f\n", (float)total_col_nz/actual_p_int);
	//refresh();
	//actual_set_sizes[actual_p_int] = 0; // so we don't add garbage to the size of the last set
	//long total_set_size = 0;
	//for (int i = 0; i < actual_p_int; i++) {
	//	if (valid_mergesets[i])
	//		total_set_size += all_sets[i].size;
	//}
	//Rprintf("mean set size: %f\n", (float)total_set_size/mergeset_count);
	//refresh();

	// let's start with one pass
	new_mergeset_count = mergeset_count;
	//#pragma omp parallel for reduction(-:new_mergeset_count) shared(valid_mergesets, actual_set_sizes)
	for (int i = 0; i < actual_p_int - 2; i += 2) {
		if (valid_mergesets[i+1] && can_merge(all_sets, i, i+1, frac_overlap_allowed)) {
			merge_sets(all_sets, i, i+1);
			//actual_set_sizes[i]++;
			valid_mergesets[i+1] = FALSE;
			new_mergeset_count--;
			//i += 2;
		} else {
			//i++;
		}
	}
	mergeset_count = new_mergeset_count;
	new_mergeset_count = mergeset_count;
	//#pragma omp parallel for reduction(-:new_mergeset_count) shared(valid_mergesets, actual_set_sizes)
	for (int i = 1; i < actual_p_int - 2; i += 2) {
		if (valid_mergesets[i+1] && valid_mergesets[i] && can_merge(all_sets, i, i+1, frac_overlap_allowed)) {
			merge_sets(all_sets, i, i+1);
			//actual_set_sizes[i] += actual_set_sizes[i+1];
			valid_mergesets[i+1] = FALSE;
			new_mergeset_count--;
			//i += 2;
		} else {
			//i++;
		}
	}
	mergeset_count = new_mergeset_count;

	//total_set_size = 0;
	//for (int i = 0; i < actual_p_int; i++) {
	//	if (valid_mergesets[i])
	//		total_set_size += all_sets[i].size;
	//}
	//Rprintf("mean set size: %f\n", (float)total_set_size/mergeset_count);
	//refresh();
	// place sets in their appropriate bin

	// indices of sets that are of particular sizes on total.
	//int set_bins_of_size[NumSets+1][actual_p_int];
	int *set_bins_of_size[NumSets+1];
	for (int i = 0; i < NumSets+1; i++)
		set_bins_of_size[i] = malloc(actual_p_int*sizeof(int));
	//int num_bins_of_size[NumSets+2];
	int *num_bins_of_size = malloc((NumSets+2)*sizeof(int));

	for (int i = 0; i <= NumSets+1; i++)
		num_bins_of_size[i] = 0;

	int count = 0;
	for (int i = 0; i < actual_p_int; i++) {
		if (valid_mergesets[i]) {
			int set_size = all_sets[i].ncols;
			if (set_size > NumSets+1)
				set_size = NumSets+1;
			valid_mergeset_indices[count] = i;
			set_bins_of_size[set_size][num_bins_of_size[set_size]] = i;
			num_bins_of_size[set_size]++;
		}
	}


	//printw("\nbins of size: ");
	//for (int i = 0; i <= NumSets+1; i++) {
	//	printw("[%d]: %d, ", i, num_bins_of_size[i]);
	//}
	//printw("\n");
	//refresh();
	int xpos, ypos;
	//getyx(stdscr, ypos, xpos);
	int *nothing = malloc(mergeset_count*sizeof(struct Beta_Set));

	//TODO: rewrite this whole section to update sets in batches, then commit
	//		the changes all at once (which sounds very openCL friendly)
	if (NumSets > 1) {
		int num_bins_to_merge = 0;
		// remove all sets of size 1
		int *sets_to_merge = malloc(actual_p_int*sizeof(int));
		memset(sets_to_merge, 0, actual_p_int*sizeof(int));

		int moved_something = 1;
		for (int iter2 = 0; iter2 < 2 && moved_something; iter2++) {
			moved_something = 0;
			for (int small_set = 1; small_set < NumSets - 1; small_set++) {
				//move(ypos, xpos);
				//printw("current state: ");
				//for (int i = 0; i <= NumSets+1; i++)
				//	printw("[%d]: %d, ", i, num_bins_of_size[i]);
				//printw("\nclearing set_size %d\n", small_set);
				//refresh();
				for (int iter = 0; iter < 5 && (iter < num_bins_of_size[small_set]); iter++) {
					//move(ypos+1, xpos);
					//printw("iter %d\n", iter);
					//refresh();
					// compare the first n columns of the `1' bin, w/ the first n of the last, to see
					// if they can be merged.
					int n;
					for (int large_set = NumSets-1; large_set > small_set; large_set--) {
						if (num_bins_of_size[small_set] < num_bins_of_size[large_set])
							n = num_bins_of_size[small_set];
						else
							n = num_bins_of_size[large_set];
						if (n == 0)
							continue;
						//TODO: don't use iter, iter+1, at least choose from a random distribution instead.
						num_bins_to_merge = compare_n(all_sets, valid_mergesets, set_bins_of_size, num_bins_of_size, sets_to_merge, small_set, large_set, n, iter, iter+50*iter2+1, frac_overlap_allowed);
						if (num_bins_to_merge > 0) {
							moved_something = 1;
							merge_n(all_sets, set_bins_of_size, num_bins_of_size, valid_mergesets, sets_to_merge, small_set,  large_set, n, iter, iter+50*iter2+1, num_bins_to_merge);
							mergeset_count -= num_bins_to_merge;
						}
					}
					// merge with same set, ensure no overlap
					n = num_bins_of_size[small_set]/2;
					if (n != 0) {
						int offset = (iter + 50*iter2)%n;
						num_bins_to_merge = compare_n(all_sets, valid_mergesets, set_bins_of_size, num_bins_of_size, sets_to_merge, small_set, small_set, n, 0 + offset, n + offset, frac_overlap_allowed);
						merge_n(all_sets, set_bins_of_size, num_bins_of_size, valid_mergesets, sets_to_merge, small_set, small_set, n, 0 + offset, n + offset, num_bins_to_merge);
						mergeset_count -= num_bins_to_merge;
					}
				}
			}
		}
	}

	//printw("\nafter: bins of size: ");
	//for (int i = 0; i <= NumSets+1; i++) {
	//	printw("[%d]: %d, ", i, num_bins_of_size[i]);
	//}
	//printw("\n");
	//refresh();

	//all_sets = remove_invalid_sets(all_sets, valid_mergesets, actual_p_int, new_mergeset_count, actual_set_sizes);

	//printw("some useful statistics:\n");
	Rprintf("mean set size: %.1f\n", (float)actual_p_int/mergeset_count);

	//TODO: only works for contiguous sets at the moment (if there)
	int set_size, cur_set = 0;
	return_sets.number_of_sets = mergeset_count;
	return_sets.sets = malloc(mergeset_count*sizeof(struct Beta_Set));
	for (int i = 0; i < actual_p_int; i++) {
		if (valid_mergesets[i] == FALSE)
			continue;
		return_sets.sets[cur_set].set_size = all_sets[i].ncols;
		return_sets.sets[cur_set].set = malloc(all_sets[i].ncols*sizeof(int));
		for (int j = 0; j < all_sets[i].ncols; j++) {
			return_sets.sets[cur_set].set[j] = all_sets[i].cols[j];
		}

		cur_set++;
	}

	free(valid_mergesets);
	free(valid_mergeset_indices);

	return return_sets;
}

Beta_Sets find_beta_sets(XMatrix_sparse x2col, int actual_p_int, int n, double frac_overlap_allowed) {
	return merge_find_beta_sets(x2col, actual_p_int, n, frac_overlap_allowed);
}

XMatrix read_x_csv(char *fn, int n, int p) {
	char *buf = NULL;
	size_t line_size = 0;
	int **X = malloc(p*sizeof(int*));
	//gsl_spmatrix *X_sparse = gsl_spmatrix_alloc(n, p);

	// forces X[...] to be sequential. (and adds some segfaults).
	//int *Xq = malloc(n*p*sizeof(int));
	//for (int i = 0; i < n; i++)
	//	X[i] = &Xq[p*i];

	for (int i = 0; i < p; i++)
		X[i] = malloc(n*sizeof(int));

	//move(1,0);
	//printw("reading X from: \"%s\"\n", fn);
	//refresh();

	FILE *fp = fopen(fn, "r");
	if (fp == NULL) {
		perror("opening failed");
	}

	int col = 0, row = 0, actual_cols = p;
	int readline_result = 0;
	while((readline_result = getline(&buf, &line_size, fp)) > 0) {
		// remove name from beginning (for the moment)
		int i = 1;
		while (buf[i] != '"')
			i++;
		i++;
		// read to the end of the line
		while (buf[i] != '\n' && i < line_size) {
			if (buf[i] == ',')
				{i++; continue;}
			//printf("setting X[%d][%d] to %c\n", row, col, buf[i]);
			if (buf[i] == '0') {
				X[col][row] = 0;
			}
			else if (buf[i] == '1') {
				X[col][row] = 1;
				//gsl_spmatrix_set(X_sparse, row, col, 1);
			}
			else {
				fprintf(stderr, "format error reading X from %s at row: %d, col: %d\n", fn, row, col);
				exit(0);
			}
			i++;
			if (++col >= p)
				break;
		}
		if (buf[i] != '\n')
			fprintf(stderr, "reached end of file without a newline\n");
		if (col < actual_cols)
			actual_cols = col;
		col = 0;
		if (++row >= n)
			break;
	}
	if (readline_result == -1)
		fprintf(stderr, "failed to read line, errno %d\n", errno);

	if (actual_cols < p) {
		printf("number of columns < p, should p have been %d?\n", actual_cols);
		p = actual_cols;
	}
	//move(2,0);
	//printw("read %dx%d, freeing stuff\n", row, actual_cols);
	//refresh();
	free(buf);
	XMatrix xmatrix;
	xmatrix.X = X;
	//xmatrix.X_sparse = gsl_spmatrix_ccs(X_sparse);
	xmatrix.actual_cols = actual_cols;
	return xmatrix;
}

double *read_y_csv(char *fn, int n) {
	char *buf = malloc(BUF_SIZE);
	char *temp = malloc(BUF_SIZE);
	memset(buf, 0, BUF_SIZE);
	double *Y = malloc(n*sizeof(double));

	//move(3,0);
	//printw("reading Y from: \"%s\"\n", fn);
	//refresh();
	FILE *fp = fopen(fn, "r");
	if (fp == NULL) {
		perror("opening failed");
	}

	int col = 0, i = 0;
	// drop the first line
	//if (fgets(buf, BUF_SIZE, fp) == NULL)
	//	fprintf(stderr, "failed to read first line of Y from \"%s\"\n", fn);
	while(fgets(buf, BUF_SIZE, fp) != NULL) {
		i = 1;
		// skip the name
		while(buf[i] != '"')
			i++;
		i++;
		if (buf[i] == ',')
			i++;
		// read the rest of the line as a float
		memset(temp, 0, BUF_SIZE);
		int j = 0;
		while(buf[i] != '\n')
			temp[j++] = buf[i++];
		Y[col] = atof(temp);
		//printf("temp '%s' set %d: %f\n", temp, col, Y[col]);
		col++;
	}

	// for comparison with implementations that normalise rather than
	// finding the intercept.
	if (NORMALISE_Y == 1) {
		printf("%d, normalising y values\n", NORMALISE_Y);
		double mean = 0.0;
		for (int i = 0; i < n; i++) {
			mean += Y[i];
		}
		mean /= n;
		for (int i = 0; i < n; i++) {
			Y[i] -= mean;
		}
	}

	//move(4,0);
	//printw("read %d lines, freeing stuff\n", col);
	//refresh();
	free(buf);
	free(temp);
	return Y;
}

// n.b.: for glmnet gamma should be lambda * [alpha=1] = lambda
double soft_threshold(double z, double gamma) {
	double abs = fabs(z);
	if (abs < gamma)
		return 0.0;
	double val = abs - gamma;
	if (signbit(z))
		return -val;
	else
		return val;
}

// separated to make profiling easier.
// TODO: this is taking most of the time, worth avoiding.
//		- has not been adjusted for on the fly X2.
double get_sump(int p, int k, int i, double *beta, int **X) {
	double sump = 0;
	for (int j = 0; j < p; j++) {
		if (j != k)
			//sump += X[i][j]?beta[j]:0.0;
			sump += X[i][j] * beta[j];
	}
	return sump;
}


//TODO: this takes far too long.
//	-could we store one row (of essentially these) instead?
int_pair get_num(int num, int p) {
	int offset = 0;
	int_pair ip;
	ip.i = -1;
	ip.j = -1;
	int num_post_permutation = gsl_permutation_get(global_permutation, num);
	for (int i = 0; i < p; i++) {
		for (int j = i; j < p; j++) {
			if (offset == num_post_permutation) {
				ip.i = i;
				ip.j = j;
				return ip;
			}
			offset++;
		}
	}
	return ip;
}

void update_max_rowsums(double new_value) {
	if (new_value < max_rowsums[NUM_MAX_ROWSUMS])
		return;

	//TODO: reasonable search algorithm.
	int i = NUM_MAX_ROWSUMS;
	for (; i > 0; i--) {
		if (new_value < max_rowsums[i])
			break;
	}

	// i is the index of the smallest value greater than our new one.
	// shift everything else down
	for (int j = i; j > NUM_MAX_ROWSUMS - 1; j++) {
		max_rowsums[j+1] = max_rowsums[j];
	}
	max_rowsums[i] = new_value;

	max_cumulative_rowsums[0] = max_rowsums[0];
	for (int i = 1; i < NUM_MAX_ROWSUMS; i++) {
		max_cumulative_rowsums[i] = max_cumulative_rowsums[i-1] + max_rowsums[i];
	}
}

// N.B. main effects are not first in the matrix, X[x][1] is the interaction between genes 0 and 1. (the main effect for gene 1 is at X[1][p])
// That is to say that k<p is not a good indication of whether we are looking at an interaction or not.
//TODO: max lambda
double update_beta_cyclic(XMatrix xmatrix, XMatrix_sparse xmatrix_sparse, double *Y, double *rowsum, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept, int USE_INT, int_pair *precalc_get_num) {
	double derivative = 0.0;
	double sumk = xmatrix_sparse.col_nz[k];
	double sumn = xmatrix_sparse.col_nz[k]*beta[k];
	double sump;
	int **X = xmatrix.X;
	//gsl_spmatrix *X_sparse = xmatrix.X_sparse;
	int pairwise_product = 0;
	int_pair ip;
	if (USE_INT) {
		//ip = get_num(k, p);
		ip = precalc_get_num[k];
	} else {
		ip.i = k;
		ip.j = k;
	}
	// From here on things should behave the same (this is set mostly for testing)
	USE_INT = 1;

	int j, row;
	//if (__builtin_expect(xmatrix_sparse.col_nz[k] > 2000, 1)) {
	#pragma omp parallel for shared(Y, xmatrix_sparse, rowsum, intercept) reduction (+:sumn) num_threads(NumCores)
	for (int e = 0; e < xmatrix_sparse.col_nz[k]; e++) {
		int i;
		// TODO: avoid unnecessary calculations for large lambda.
		i = xmatrix_sparse.col_nz_indices[k][e];
		// TODO: assumes X is binary
		sumn += Y[i] - intercept - rowsum[i];
	}
	//} else {
	//	for (int e = 0; e < xmatrix_sparse.col_nz[k]; e++) {
	//		// TODO: avoid unnecessary calculations for large lambda.
	//		i = xmatrix_sparse.col_nz_indices[k][e];
	//		// TODO: assumes X is binary
	//		sumn += Y[i] - intercept - rowsum[i];
	//	}
	//}
	//total_updates++;
	//total_updates_entries += xmatrix_sparse.col_nz[k];

	// TODO: This is probably slower than necessary.
	double Bk_diff = beta[k];
	if (sumk == 0.0) {
		beta[k] = 0.0;
	} else {
		beta[k] = soft_threshold(sumn, lambda*n/2)/sumk;
	}
	Bk_diff = beta[k] - Bk_diff;
	// update every rowsum[i] w/ effects of beta change.
	if (Bk_diff != 0) {
		haschanged = 1;
		for (int e = 0; e < xmatrix_sparse.col_nz[k]; e++) {
			int i = xmatrix_sparse.col_nz_indices[k][e];
			rowsum[i] += Bk_diff;
			update_max_rowsums(rowsum[i]);
		}
	} else {
		zero_updates++;
		zero_updates_entries += xmatrix_sparse.col_nz[k];
	}


	Bk_diff *= Bk_diff;
	if (Bk_diff > dBMax)
		dBMax = Bk_diff;
	return dBMax;
}

void update_beta_shoot() {
}

double update_intercept_cyclic(double intercept, int **X, double *Y, double *beta, int n, int p) {
	double new_intercept = 0.0;
	double sumn = 0.0, sumx = 0.0;

	for (int i = 0; i < n; i++) {
		sumx = 0.0;
		for (int j = 0; j < p; j++) {
			sumx += X[i][j] * beta[j];
		}
		sumn += Y[i] - sumx;
	}
	new_intercept = sumn / n;
	return new_intercept;
}

double update_beta_greedy_l1(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax) {
	double derivative = 0.0;
	double sumk = 0.0;
	double sumn = 0.0;
	double sump;

	for (int i = 0; i < n; i++) {
		sump = 0.0;
		for (int j = 0; j < p; j++) {
			// if j != k ?
				//sump += X[i][j] * beta[j];
				sump += X[i][j]?beta[j]:0.0;
		}
		//sumn += (Y[i] - sump)*(double)X[i][k];
		sumn += X[i][k]?(Y[i] - sump):0.0;
		sumk += X[i][k] * X[i][k];
	}
	derivative = -sumn;

	// TODO: This is probably slower than necessary.
	double Bkn = fmin(0.0, beta[k] - (derivative - lambda)/(sumk));
	double Bkp = fmax(0.0, beta[k] - (derivative + lambda)/(sumk));
	double Bk_diff = beta[k];
	if (Bkn < 0.0)
		beta[k] = Bkn;
	else if (Bkp > 0.0)
		beta[k] = Bkp;
	else {
		beta[k] = 0.0;
	}
	Bk_diff = fabs(beta[k] - Bk_diff);
	Bk_diff *= Bk_diff;
	if (Bk_diff > dBMax)
		dBMax = Bk_diff;
	return dBMax;
}

int worth_updating(double *col_ysum, XMatrix_sparse X2, int k, int n, int lambda) {
	if (fabs(col_ysum[k] - max_cumulative_rowsums[min(X2.col_nz[k], NUM_MAX_ROWSUMS - 1)]) > n*lambda/2) {
		return TRUE;
	}
	return FALSE;
}

/* Edgeworths's algorithm:
 * \mu is zero for the moment, since the intercept (where no effects occurred)
 * would have no effect on fitness, so 1x survived. log(1) = 0.
 * This is probably assuming that the population doesn't grow, which we may
 * not want.
 * TODO: add an intercept
 * TODO: haschanged can only have an effect if an entire iteration does nothing. This should never happen.
 */
double *simple_coordinate_descent_lasso(XMatrix xmatrix, double *Y, int n, int p, double lambda, char *method, int max_iter, int USE_INT, int verbose, double frac_overlap_allowed) {
	// TODO: until converged
		// TODO: for each main effect x_i or interaction x_ij
			// TODO: choose index i to update uniformly at random
			// TODO: update x_i in the direction -(dF(x)/de_i / B)
	//TODO: free
	VERBOSE = verbose;
	int_pair *precalc_get_num;
	int **X = xmatrix.X;
	//gsl_spmatrix *X_sparse = xmatrix.X_sparse;
	N = n;

	//move(7,0);
	//Rprintf("calculating sparse interaction matrix (cols): \n");
	//refresh();
	XMatrix_sparse X2 = sparse_X2_from_X(X, n, p, USE_INT, TRUE);
	//printw("calculating sparse interaction matrix (rows): \n");
	//XMatrix_sparse_row X2row = sparse_horizontal_X2_from_X(X, n, p, USE_INT);

	for (int i = 0; i < NUM_MAX_ROWSUMS; i++) {
		max_rowsums[i] = 0;
		max_cumulative_rowsums[i] = 0;
	}

	int p_int = p*(p+1)/2;
	double *beta;
	if (USE_INT) {
		beta = malloc(p_int*sizeof(double)); // probably too big in most cases.
		memset(beta, 0, p_int*sizeof(double));
	}
	else {
		beta = malloc(p*sizeof(double)); // probably too big in most cases.
		memset(beta, 0, p*sizeof(double));
	}
	if (USE_INT) {
		precalc_get_num = malloc(p_int*sizeof(int_pair));
		int offset = 0;
		for (int i = 0; i < p; i++) {
			for (int j = i; j < p; j++) {
				precalc_get_num[gsl_permutation_get(global_permutation_inverse,offset)].i = i;
				precalc_get_num[gsl_permutation_get(global_permutation_inverse,offset)].j = j;
				offset++;
			}
		}
	} else {
		p_int = p;
	}

	//int skip_count = 0;
	//int skip_entire_column[p_int];
	//for (int i = 0; i < p; i++)
	//	for (int j = 0; j < i; j++) {
	//		int skip_this = 1;
	//		for (int k = 0; k < n; k++) {
	//			if (X[k][i] != 0 && X[k][j] != 0) {
	//				skip_this = 0;
	//			}
	//		}
	//		if (skip_this == 1) {
	//			skip_count++;
	//			skip_entire_column[i*p + j] = 1;
	//		}
	//	}
	//printf("should skip %d columns\n", skip_count);

	double error = DBL_MAX, prev_error;
	double intercept = 0.0;
	double iter_lambda;
	int use_cyclic = 0, use_greedy = 0;

	//printw("original lambda: %f n: %d ", lambda, n);
	//lambda = lambda;
	//printw("effective lambda is %f\n", lambda);

	//move(8,0);
	if (strcmp(method,"cyclic") == 0) {
		//printw("using cyclic descent\n");
		use_cyclic = 1;
	} else if (strcmp(method, "greedy") == 0) {
		//printw("using greedy descent\n");
		use_greedy = 1;
	}
	//refresh();

	if (use_greedy == 0 && use_cyclic == 0) {
		fprintf(stderr, "exactly one of cyclic/greedy must be specified\n");
		return NULL;
	}

	// initially every value will be 0, since all betas are 0.
	double rowsum[n];
	memset(rowsum, 0, n*sizeof(double));

	colsum = malloc(p_int*sizeof(double));
	memset(colsum, 0, p_int*sizeof(double));

	col_ysum = malloc(p_int*sizeof(double));
	for (int col = 0; col < p_int; col++) {
		for (int row_ind = 0; row_ind < X2.col_nz[col]; row_ind++) {
			col_ysum[col] += Y[X2.col_nz_indices[col][row_ind]];
		}
	}

	// find largest number of non-zeros in any column
	int largest_col = 0;
	long total_col = 0;
	for (int i = 0; i < p_int; i++) {
		if (X2.col_nz[i] > largest_col) {
			largest_col = X2.col_nz[i];
		}
		total_col += X2.col_nz[i];
	}
	int main_sum = 0;
	for (int i = 0; i < p; i++)
		for (int j = 0; j < n; j++)
			main_sum += X[i][j];
	//move(9,0);
	//printw("\nlargest column has %d non-zero entries (out of %d)\n", largest_col, n);
	//move(10,0);
	//printw("mean column has %f (%f main) non-zero entries (out of %d)\n", (double)total_col/p_int, (double)main_sum/p, n);
	//refresh();

#ifdef LIMIT_OVERLAP
	//printw("finding simultaneously updateable beta sets... ");
	//refresh();
	Beta_Sets beta_sets;
	if (USE_INT == 1)
		beta_sets = find_beta_sets(X2, p_int, n, frac_overlap_allowed);
	else
		beta_sets = find_beta_sets(X2, p, n, frac_overlap_allowed);
	//printw(" done\n");
	//refresh();
#endif
	int scrx, scry;
	//getyx(stdscr, scry, scrx);

	struct timespec start, end;
	double cpu_time_used;

	int *cols_to_update = malloc(p_int*sizeof(int));
	clock_gettime(CLOCK_REALTIME, &start);
	//TODO: make ratio an option
	double final_lambda = 0.1*lambda;
	Rprintf("running from lambda %.2f to lambda %.2f\n", lambda, final_lambda);
	int lambda_count = 1;
	for (int iter = 0; iter < max_iter && lambda > final_lambda; iter++) {
		//refresh();
		prev_error = error;
		error = 0;
		double dBMax = 0.0; // largest beta diff this cycle

		// update intercept (don't for the moment, it should be 0 anyway)
		//intercept = update_intercept_cyclic(intercept, X, Y, beta, n, p);
		//iter_lambda = lambda*(max_iter-iter)/max_iter;
		//printf("using lambda = %f\n", iter_lambda);

		haschanged = 1;
		int count=5;
		//#pragma omp parallel for num_threads(1) reduction(+:count) // >1 threads will (unsurprisingly) lead to inconsistent (& not reproducable) results
		//for (int k = 0; k < p_int; k++) {
		//	if (k % (p_int/100) == 0) {
		//		move(12,0);
		//		printw("iteration %d: ", iter);
		//		refresh();
		//		move(12,15);
		//		printw("%d%%", count++);
		//		refresh();
		//	}

			// update the predictor \Beta_k
#ifdef LIMIT_OVERLAP
			#pragma omp parallel num_threads(NumCores) shared(col_ysum, xmatrix, X2, Y, rowsum, beta, precalc_get_num) reduction(+:total_updates, skipped_updates, skipped_updates_entries, total_updates_entries) //schedule(static, 1)
			for (int i = 0; i <  beta_sets.number_of_sets; i++) {
				#pragma omp for 
				for (int j = 0; j < beta_sets.sets[i].set_size; j++) {
					int k = beta_sets.sets[i].set[j];
#else
				#pragma omp parallel for num_threads(NumCores) shared(col_ysum, xmatrix, X2, Y, rowsum, beta, precalc_get_num) reduction(+:total_updates, skipped_updates, skipped_updates_entries, total_updates_entries) //schedule(static, 1)
				for (int k = 0; k < p_int; k++) {

#endif
					if (worth_updating(col_ysum, X2, k, n, lambda)) {
						dBMax = update_beta_cyclic(xmatrix, X2, Y, rowsum, n, p, lambda, beta, k, dBMax, intercept, USE_INT, precalc_get_num);
						total_updates++;
						total_updates_entries += X2.col_nz[k];
					}
					else {
						skipped_updates++;
						total_updates++;
						skipped_updates_entries += X2.col_nz[k];
						total_updates_entries += X2.col_nz[k];
					}
				}
#ifdef LIMIT_OVERLAP
			}
#endif
		//}
		haschanged = 0;
		//move(scry, scrx);
		//printw("\n\n");

		error = 0;
		// caculate cumulative error after update
		if (USE_INT == 0)
			for (int row = 0; row < n; row++) {
				double sum = 0;
				for (int k = 0; k < p; k++) {
					sum += X[k][row]*beta[k];
				}
				double e_diff = Y[row] - intercept - sum;
				e_diff *= e_diff;
				error += e_diff;
			}
		else {
			double *row_err_sums = malloc(n*sizeof(double));
			memset(row_err_sums, 0, n*sizeof(double));
			for (int col = 0; col < p_int; col++) {
				double sum = 0;
				for (int row_ind = 0; row_ind < X2.col_nz[col]; row_ind++) {
					row_err_sums[X2.col_nz_indices[col][row_ind]] += beta[col];
				}
			}

			for (int row = 0; row < n; row++) {
				double row_err = Y[row] - intercept - row_err_sums[row];
				if (row_err_sums[row] < -0.1) {
//					printf("row %d, Y: %f err: %f\n", row, Y[row], row_err_sums[row]);
				}
				error += row_err*row_err;
			}

			free(row_err_sums);
		}
		error /= n;
		//Rprintf("mean squared error is now %f, w/ intercept %f\n", error, intercept);
		//Rprintf("error diff: %.2f\%\n", prev_error/error);
		//printw("indices significantly negative (-500):\n");
		//int printed = 0;
		////TODO: remove hack to prevent printing too many for the terminal
		//for (int i = 0; i < p_int && printed < 10; i++) {
		//	if (beta[i] < -500) {
		//		printed++;
		//		int_pair ip = get_num(i, p);
		//		if (ip.i == ip.j)
		//			printw("main: %d (%d):\t\t\t %f\n", i, ip.i + 1, beta[i]);
		//		else
		//			printw("int: %d  (%d, %d):\t\t %f\n", i, ip.i + 1, ip.j + 1, beta[i]);
		//	}
		//}
		// Be sure to clean up anything extra we allocate
		// TODO: don't actually do this, see glmnet convergence conditions for a more detailed approach.
		if (prev_error/error < HALT_BETA_DIFF) {
			//Rprintf("largest change (%f) was less than %f, halting after %d iterations\n", prev_error/error, HALT_BETA_DIFF, iter + 1);
			Rprintf("done lambda %d after %d iterations, final error %.1f\n", lambda_count, iter + 1, error);
			lambda_count++;
			lambda *= 0.9;
		}

		//printw("done iteration %d\n", iter);
		//clrtobot();
	}

	clock_gettime(CLOCK_REALTIME, &end);
	cpu_time_used = ((double)(end.tv_nsec-start.tv_nsec))/1e9 + (end.tv_sec - start.tv_sec);

	Rprintf("lasso done in %.4f seconds, columns skipped %ld out of %ld a.k.a (%f%%)\n", cpu_time_used, skipped_updates, total_updates, (skipped_updates*100.0)/((long)total_updates));
	Rprintf("cols: performed %d zero updates (%f%%)\n", zero_updates, ((float)zero_updates/(total_updates)) * 100);
	Rprintf("skipped entries %ld out of %ld a.k.a (%f%%)\n", skipped_updates_entries, total_updates_entries, (skipped_updates_entries*100.0)/((long)total_updates_entries));
	free(precalc_get_num);
	Rprintf("entries: performed %d zero updates (%f%%)\n", zero_updates_entries, ((float)zero_updates_entries/(total_updates_entries)) * 100);

	return beta;
}

int **X2_from_X(int **X, int n, int p) {
	int **X2 = malloc(n*sizeof(int*));
	for (int row = 0; row < n; row++) {
		X2[row] = malloc(((p*(p+1))/2)*sizeof(int));
		int offset = 0;
		for (int i = 0; i < p; i++) {
			for (int j = i; j < p; j++) {
				X2[row][offset++] = X[row][i] * X[row][j];
			}
		}
	}
	return X2;
}


// TODO: write a test comparing this to non-sparse X2
XMatrix_sparse sparse_X2_from_X(int **X, int n, int p, int USE_INT, int shuffle) {
	XMatrix_sparse X2;
	int colno, val, length;
	int p_int = (p*(p+1))/2;
	int iter_done = 0;
	int actual_p_int = 0;

	if (!USE_INT) {
		X2.col_nz_indices = malloc(p*sizeof(int *));
		X2.col_nz = malloc(p*sizeof(int));
		actual_p_int = p;
	} else {
		X2.col_nz_indices = malloc(p_int*sizeof(int *));
		X2.col_nz = malloc(p_int*sizeof(int));
		actual_p_int = p_int;
	}

	//TODO: iter_done isn't exactly being updated safely
	#pragma omp parallel for shared(X2, X, iter_done) private(length, val, colno) num_threads(8)
	for (int i = 0; i < p; i++) {
		for (int j = i; j < p; j++) {
			GSList *current_col = NULL;
			// only include main effects (where i==j) unless USE_INT is set.
			if (USE_INT || j == i) {
				if (USE_INT)
					// worked out by hand as being equivalent to the offset we would have reached.
					colno = (2*(p-1) + 2*(p-1)*(i-1) - (i-1)*(i-1) - (i-1))/2 + j;
				else
					colno = i;

				for (int row = 0; row < n; row++) {
					val = X[i][row] * X[j][row];
					if (val == 1) {
						current_col = g_slist_prepend(current_col, (void*)(long)row);
					}
					else if (val != 0)
						fprintf(stderr, "Attempted to convert a non-binary matrix, values will be missing!\n");
				}
				length = g_slist_length(current_col);
				current_col = g_slist_reverse(current_col);

				X2.col_nz_indices[colno] = malloc(length*sizeof(int));
				X2.col_nz[colno] = length;

				GSList *current_col_ind = current_col;
				int temp_counter = 0;
				while(current_col_ind != NULL) {
					X2.col_nz_indices[colno][temp_counter++] = (int)(long)current_col_ind->data;
					current_col_ind = current_col_ind->next;
				}

				g_slist_free(current_col);
				current_col = NULL;
			}
		}
		iter_done++;
		//if (omp_get_thread_num() == 0) {
		//	move(7,48);
		//	printw("%.1f%%\n", (float)iter_done*100/p);
		//	refresh();
		//}
	}

	//int shuffle_order[p_int];
	gsl_rng *r;
	gsl_permutation *permutation = gsl_permutation_alloc(actual_p_int);
	gsl_permutation_init(permutation);
	gsl_rng_env_setup();
	const gsl_rng_type *T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	if (shuffle == TRUE)
		gsl_ran_shuffle(r, permutation->data, actual_p_int, sizeof(size_t));
	int **permuted_indices = malloc(actual_p_int * sizeof(int*));
	int *permuted_nz = malloc(actual_p_int * sizeof(int));
	for (int i = 0; i < actual_p_int; i++) {
		permuted_indices[i] = X2.col_nz_indices[permutation->data[i]];
		permuted_nz[i] = X2.col_nz[permutation->data[i]];
	}
	free(X2.col_nz_indices);
	free(X2.col_nz);
	free(r);
	X2.col_nz_indices = permuted_indices;
	X2.col_nz = permuted_nz;
	X2.permutation = permutation;
	global_permutation = permutation;
	global_permutation_inverse = gsl_permutation_alloc(permutation->size);
	gsl_permutation_inverse(global_permutation_inverse, permutation);

	return X2;
}

//TODO: sparse row matrix (for interaction counts)
XMatrix_sparse_row sparse_horizontal_X2_from_X(int **X, int n, int p, int USE_INT) {
	XMatrix_sparse_row X2;
	int rowno, val, length, colno;
	int p_int = (p*(p+1))/2;
	int iter_done = 0;

	X2.row_nz_indices = malloc(n*sizeof(int *));
	X2.row_nz = malloc(n*sizeof(int));

	#pragma omp parallel for shared(X2, X) private(length, val, colno) shared(iter_done)
	for (int rowno = 0; rowno < n; rowno++) {
		GSList *current_row = NULL;
		// only include main effects (where i==j) unless USE_INT is set.
		for (int i = 0; i < p; i++) {
			for (int j = i; j < p; j++) {
				// only include main effects (where i==j) unless USE_INT is set.
				if (USE_INT || j == i) {
					if (USE_INT)
						// worked out by hand as being equivalent to the offset we would have reached.
						colno = (2*(p-1) + 2*(p-1)*(i-1) - (i-1)*(i-1) - (i-1))/2 + j;
					else
						colno = i;
					if (X[i][rowno] * X[j][rowno] == 1) {
						current_row = g_slist_prepend(current_row, (void*)(long)colno);
					}

				}
			}
		}
		length = g_slist_length(current_row);
		current_row = g_slist_reverse(current_row);

		X2.row_nz_indices[rowno] = malloc(length*sizeof(int));
		X2.row_nz[rowno] = length;

		GSList *current_row_ind = current_row;
		int temp_counter = 0;
		while(current_row_ind != NULL) {
			X2.row_nz_indices[rowno][temp_counter++] = (int)(long)current_row_ind->data;
			current_row_ind = current_row_ind->next;
		}

		g_slist_free(current_row);
		current_row = NULL;
		iter_done += 1;
		//if (omp_get_thread_num() == 0) {
		//	move(8,48);
		//	printw("%.1f%%\n", (float)iter_done*100/n);
		//	refresh();
		//}
	}
	return X2;
}
