#include "liblasso.h"
#include <cstdio>

static long log_file_offset;
static int int_print_len;

// print to log: metadata required to resume from the log
FILE* init_log(char* filename, int n, int p, int num_betas, char** job_args,
    int job_args_num)
{
    int_print_len = std::log10(p*p*p) + 1;
    FILE* log_file = fopen(filename, "w+");
    fprintf(log_file, "still running\n");
    for (int i = 0; i < job_args_num; i++) {
        fprintf(log_file, "%s ", job_args[i]);
    }
    fprintf(log_file, "\n");
    fprintf(log_file, "lasso log file. metadata follows on the next few lines.\n \
The remaining log is {[ ]done/[w]ip} $current_iter, $current_lambda\\n $beta_1, $beta_2 ... $beta_k");
    fprintf(log_file, "num_rows, num_cols\n");
    fprintf(log_file, "%d, %d\n", n, p);
    log_file_offset = ftell(log_file);
    return log_file;
}

static int log_pos = 0;
// save the current beta values to a log, so the program can be resumed if it is
// interrupted
void save_log(int iter, float lambda_value, int lambda_count, Beta_Value_Sets* beta_sets,
    FILE* log_file)
{
    int real_n_betas = beta_sets->beta1.size() + beta_sets->beta2.size() + beta_sets->beta3.size();
    // Rather than filling the log with beta values, we want to only keep two
    // copies. The current one, and a backup in case we stop while writing the
    // current one.
    // returns to the begginning if we wrote to the end last time.
    if (log_pos % 2 == 0) {
        fseek(log_file, log_file_offset, SEEK_SET);
    }
    log_pos = (log_pos + 1) % 2;

    // indicate that this entry is a work in progress
    long log_entry_start_pos = ftell(log_file);
    fprintf(log_file, "w");

    // write num beta1/2/3 values
    fprintf(log_file, "%.*d, %.*d, %.*d\n", int_print_len, beta_sets->beta1.size(), int_print_len, beta_sets->beta2.size(), int_print_len, beta_sets->beta3.size());

    // print beta 1/2/3 values each on a new line.
    fprintf(log_file, "%.*d, %.*d, %+.6e\n", int_print_len, iter, int_print_len, lambda_count, lambda_value);
    for (auto it = beta_sets->beta1.begin(); it != beta_sets->beta1.end(); it++) {
        fprintf(log_file, "%.*d,%+.6e, ", int_print_len, it->first, it->second);
    }
    fprintf(log_file, "\n");
    for (auto it = beta_sets->beta2.begin(); it != beta_sets->beta2.end(); it++) {
        fprintf(log_file, "%.*d,%+.6e, ", int_print_len, it->first, it->second);
    }
    fprintf(log_file, "\n");
    for (auto it = beta_sets->beta3.begin(); it != beta_sets->beta3.end(); it++) {
        fprintf(log_file, "%.*d,%+.6e, ", int_print_len, it->first, it->second);
    }
    fprintf(log_file, "\n");

    long log_entry_end_pos = ftell(log_file);
    fseek(log_file, log_entry_start_pos, SEEK_SET);
    fprintf(log_file, " ");
    fseek(log_file, log_entry_end_pos, SEEK_SET);
}

void close_log(FILE* log_file)
{
    fseek(log_file, 0, SEEK_SET);
    fprintf(log_file, "finished     \n");
    fclose(log_file);
}

int check_can_restore_from_log(char* filename, int n, int p, int num_betas,
    char** job_args, int job_args_num)
{
    int buf_size = num_betas * 16 + 500;
    int can_use = FALSE;
    FILE* log_file = fopen(filename, "r");
    if (log_file == NULL) {
        return FALSE;
    }
    char* our_args = malloc(500);
    char* buffer = malloc(buf_size);

    memset(our_args, 0, sizeof(our_args));
    for (int i = 0; i < job_args_num; i++) {
        sprintf(our_args + strlen(our_args), "%s ", job_args[i]);
    }
    sprintf(our_args + strlen(our_args), "\n");

    // printf("checking log\n");
    fgets(buffer, buf_size, log_file);
    // printf("comparing '%s', '%s'n", buffer, "still running");
    if (strcmp(buffer, "still running\n") == 0) {
        // there was an interrupted run, we should check if it was this one.
        fgets(buffer, buf_size, log_file);
        // printf("comparing '%s', '%s'\n", buffer, our_args);
        if (strcmp(buffer, our_args) == 0) {
            // the files were the same!
            can_use = TRUE;
        }
    }

    free(buffer);
    free(our_args);
    fclose(log_file);
    return can_use;
}

// returns the opened log for future use.
FILE* restore_from_log(char* filename, int n, int p, int num_betas,
    char** job_args, int job_args_num, int* actual_iter,
    int* actual_lambda_count, float* actual_lambda_value,
    robin_hood::unordered_flat_map<long, float> actual_beta)
{

    FILE* log_file = fopen(filename, "r+");
    int buf_size = num_betas * 16 + 500;
    char* buffer = malloc(buf_size);
    Rprintf("restoring from log\n");

    // (none of this actually changes, we just need to set the log_file_offset)
    fprintf(log_file, "still running\n");
    for (int i = 0; i < job_args_num; i++) {
        fprintf(log_file, "%s ", job_args[i]);
    }
    fprintf(log_file, "\n");
    fprintf(log_file, "lasso log file. metadata follows on the next few lines.\n \
The remaining log is {[ ]done/[w]ip} $current_iter, $current_lambda\\n $beta_1, $beta_2 ... $beta_k");
    fprintf(log_file, "num_rows, num_cols, num_betas\n");
    fprintf(log_file, "%d, %d, %d\n", n, p, num_betas);
    log_file_offset = ftell(log_file);

    // now we're at the first saved line, check whether it's a complete
    // checkpoint.
    int can_restore = TRUE;
    long first_pos = ftell(log_file);
    fgets(buffer, buf_size, log_file);
    printf("first buf: '%s'\n", buffer);
    if (strncmp(buffer, "w", 1) == 0) {
        printf("first entry unusable\n");
        // line is incomplete, don't use it.
        fgets(buffer, buf_size, log_file);
        fgets(buffer, buf_size, log_file);
        printf("second buf: '%s'\n", buffer);
        if (strncmp(buffer, "w", 1) == 0) {
            printf("second entry unusable\n");
            // so is the other one, don't change anything.
            can_restore = FALSE;
            Rprintf(
                "warning: failed to restore from log, all entries were invalid.\n");
        }
    } else {
        // the first one was fine, but the second one may be more recent.
        int first_iter, first_lambda_count;
        int second_iter, second_lambda_count;
        float first_lambda_value;
        float second_lambda_value;
        sscanf(buffer, " %d, %d, %le\n", &first_iter, &first_lambda_count,
            &first_lambda_value);
        fgets(buffer, buf_size, log_file);
        fgets(buffer, buf_size, log_file);
        long second_pos = ftell(log_file);
        sscanf(buffer, " %d, %d, %le\n", &second_iter, &second_lambda_count,
            &second_lambda_value);
        printf("first_lambda_count: %d\n", first_lambda_count);
        printf("second_lambda_count: %d\n", second_lambda_count);
        if (strncmp(buffer, "w", 1) == 0 || first_lambda_count > second_lambda_count || (first_lambda_count == second_lambda_count && first_iter > second_iter)) {
            printf("first entry > second_entry\n");
            // but we can't/shouldn't use this one, go back to the first
            fseek(log_file, first_pos, SEEK_SET);
            fgets(buffer, buf_size, log_file);
        }
    }
    if (can_restore) {
        // buffer contains the current lambda and iter values.
        int first_iter = -1, first_lambda_count = -1;
        float first_lambda_value = -1;
        printf("final buf: '%s'\n", buffer);
        sscanf(buffer, " %d, %d, %le", &first_iter, &first_lambda_count,
            &first_lambda_value);
        printf("%d, %d, %f\n", first_iter, first_lambda_count, first_lambda_value);
        sscanf(buffer, " %d, %d, %le\n", actual_iter, actual_lambda_count,
            actual_lambda_value);
        printf("lambda_count is now %d, lambda is now %f, iter is now %d\n",
            *actual_lambda_count, *actual_lambda_value, *actual_iter);
        // we actually only need the beta values, which are on the current line.
        fgets(buffer, buf_size, log_file);

        // printf("values_buf: '%s'\n", buffer);
        long offset = 0;
        for (int i = 0; i < num_betas; i++) {
            // printf("values_buf: '%s'\n", buffer + offset);
            // printf("reading beta %d\n", i);
            actual_beta[i] = 0.0;
            int ret = sscanf(buffer + offset, "%le, ", &actual_beta[i]);
            if (ret != 1) {
                printf("failed to match value in log, bad things will now happen\n");
            }
            // printf("value %lf\n", actual_beta[i]);
            offset += 15;
        }
    }

    free(buffer);
    Rprintf("done restoring from log\n");
    return log_file;
}