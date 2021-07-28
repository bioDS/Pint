#include "liblasso.h"
#include <cstdio>
#include <cstdlib>

static long log_file_offset;
static int int_print_len;

void init_print(FILE *log_file, long n, long p, long num_betas, const char** job_args,
    long job_args_num)
{
    int_print_len = std::log10(p*p*p) + 1;
    fprintf(log_file, "still running\n");
    for (long i = 0; i < job_args_num; i++) {
        fprintf(log_file, "%s ", job_args[i]);
    }
    fprintf(log_file, "\n");
    fprintf(log_file, "lasso log file. metadata follows on the next few lines.\n \
The remaining log is {[ ]done/[w]ip} $current_iter, $current_lambda\\n $beta_1, $beta_2 ... $beta_k");
    fprintf(log_file, "num_rows, num_cols\n");
    fprintf(log_file, "%ld, %ld\n", n, p);
    log_file_offset = ftell(log_file);
}


// print to log: metadata required to resume from the log
FILE* init_log(const char* filename, long n, long p, long num_betas, const char** job_args,
    long job_args_num)
{
    FILE* log_file = fopen(filename, "w+");
    init_print(log_file, n, p, num_betas, job_args, job_args_num);
    return log_file;
}

static long log_pos = 0;
// save the current beta values to a log, so the program can be resumed if it is
// interrupted
void save_log(long iter, float lambda_value, long lambda_count, Beta_Value_Sets* beta_sets,
    FILE* log_file)
{
    long real_n_betas = beta_sets->beta1.size() + beta_sets->beta2.size() + beta_sets->beta3.size();
    // Rather than filling the log with beta values, we want to only keep two
    // copies. The current one, and a backup in case we stop while writing the
    // current one.
    // returns to the begginning if we wrote to the end last time.
    bool use_first_entry = false;
    if (log_pos % 2 == 0) {
        fseek(log_file, log_file_offset, SEEK_SET);
        use_first_entry = true;
    }
    log_pos = (log_pos + 1) % 2;

    // indicate that this entry is a work in progress
    long log_entry_start_pos = ftell(log_file);
    fprintf(log_file, "w");

    // write num beta1/2/3 values
    fprintf(log_file, "%.*ld, %.*ld, %.*ld\n", int_print_len, beta_sets->beta1.size(), int_print_len, beta_sets->beta2.size(), int_print_len, beta_sets->beta3.size());

    // print beta 1/2/3 values each on a new line.
    fprintf(log_file, "%.*ld, %.*ld, %+.6e\n", int_print_len, iter, int_print_len, lambda_count, lambda_value);
    for (auto it = beta_sets->beta1.begin(); it != beta_sets->beta1.end(); it++) {
        fprintf(log_file, "%.*ld,%+.6e, ", int_print_len, it->first, it->second);
    }
    fprintf(log_file, "\n");
    for (auto it = beta_sets->beta2.begin(); it != beta_sets->beta2.end(); it++) {
        fprintf(log_file, "%.*ld,%+.6e, ", int_print_len, it->first, it->second);
    }
    fprintf(log_file, "\n");
    for (auto it = beta_sets->beta3.begin(); it != beta_sets->beta3.end(); it++) {
        fprintf(log_file, "%.*ld,%+.6e, ", int_print_len, it->first, it->second);
    }
    fprintf(log_file, "\n");

    long log_entry_end_pos = ftell(log_file);
    // make sure second entry is invalid (we might have overwritten bits of it)
    if (use_first_entry)
        fprintf(log_file, "w");
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

long check_can_restore_from_log(const char* filename, long n, long p, long num_betas,
    const char** job_args, long job_args_num)
{
    long buf_size = num_betas * 16 + 500;
    long can_use = FALSE;
    FILE* log_file = fopen(filename, "r");
    if (log_file == NULL) {
        return FALSE;
    }
    char* our_args = (char*)malloc(500);
    char* buffer = (char*)malloc(buf_size);

    memset(our_args, 0, sizeof(our_args));
    for (long i = 0; i < job_args_num; i++) {
        sprintf(our_args + strlen(our_args), "%s ", job_args[i]);
    }
    sprintf(our_args + strlen(our_args), "\n");

    // printf("checking log\n");
    char* ret = fgets(buffer, buf_size, log_file);
    // printf("comparing '%s', '%s'n", buffer, "still running");
    if (strcmp(buffer, "still running\n") == 0) {
        // there was an interrupted run, we should check if it was this one.
        ret = fgets(buffer, buf_size, log_file);
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
FILE* restore_from_log(const char* filename, bool check_args, long n, long p, 
    const char** job_args, long job_args_num, long* actual_iter,
    long* actual_lambda_count, float* actual_lambda_value,
    Beta_Value_Sets *actual_beta_sets)
{

    long num_betas = 100;
    FILE* log_file = fopen(filename, "r+");
    long buf_size = num_betas * (16+int_print_len) + 500;
    char* buffer = new char[buf_size];
    long beta1_size = -1, beta2_size = -1, beta3_size = -1;
    Rprintf("restoring from log\n");

    // the first 5 rows are commentary & X size.
    char* ret = fgets(buffer, buf_size, log_file);
    ret = fgets(buffer, buf_size, log_file);
    ret = fgets(buffer, buf_size, log_file);
    ret = fgets(buffer, buf_size, log_file);
    ret = fgets(buffer, buf_size, log_file); // contains n, p
    sscanf(buffer, "%ld, %ld", &n, &p);
    int_print_len = std::log10(p*p*p) + 1;
    log_file_offset = ftell(log_file);
    // init_print(log_file, n, p, num_betas, job_args, job_args_num);

    // now we're at the first saved line, check whether it's a complete
    // checkpoint.
    long can_restore = TRUE;
    long first_pos = ftell(log_file);

    // assumes the line has already been read into buffer.
    auto read_beta_sizes = [&]() {
        // buffer contains ' %ld, %ld, %ld' [entries in row 1,2,3]
        sscanf(buffer, " %ld, %ld, %ld\n", &beta1_size, &beta2_size, &beta3_size);
        printf(": beta sizes: %ld, %ld, %ld\n", beta1_size, beta2_size, beta3_size);
        long max_size = std::max(beta1_size, std::max(beta2_size, beta3_size))*(16+int_print_len);
        if (max_size > buf_size) {
            printf("setting new buf size for %ld entries\n", max_size);
            buf_size = max_size;
            buffer = (char*)realloc(buffer, buf_size*(16+int_print_len) + 500);
        }
        printf(": using buf size: %ld\n", buf_size);
    };

    auto skip_entries = [&]() {
        ret = fgets(buffer, buf_size, log_file);
        ret = fgets(buffer, buf_size, log_file);
        ret = fgets(buffer, buf_size, log_file);
    };

    ret = fgets(buffer, buf_size, log_file);
    printf("first buf: '%s'\n", buffer);
    if (strncmp(buffer, "w", 1) == 0) {
        printf("first entry unusable\n");
        // line is incomplete, don't use it.
        // just to read past it, we want to get the max size of each row.
        read_beta_sizes();
        ret = fgets(buffer, buf_size, log_file);

        // skip over the lambda details and the entries;
        skip_entries();

        printf("second buf: '%s'\n", buffer);
        if (strncmp(buffer, "w", 1) == 0) {
            printf("second entry unusable\n");
            // so is the other one, don't change anything.
            can_restore = FALSE;
            Rprintf(
                "warning: failed to restore from log, all entries were invalid.\n");
        }
    } else {
        printf(": first entry was usable\n");
        read_beta_sizes();
        ret = fgets(buffer, buf_size, log_file); // read lambda/iter info.
        // the first one was fine, but the second one may be more recent.
        long first_iter, first_lambda_count;
        long second_iter, second_lambda_count;
        float first_lambda_value;
        float second_lambda_value;
        sscanf(buffer, "%ld, %ld, %e\n", &first_iter, &first_lambda_count,
            &first_lambda_value);
        printf(": first_iter, lambda_count, lambda_value: %ld,%ld,%f\n", first_iter, first_lambda_count, first_lambda_value);

        skip_entries();
        ret = fgets(buffer, buf_size, log_file); //there's a newline between the two
        long second_pos = ftell(log_file);
        ret = fgets(buffer, buf_size, log_file);

        // check the second entry, but only if it exists.
        if (!feof(log_file)) {
            printf("second entry exists\n");
            if (strncmp(buffer, "w", 1) == 0) {
                printf("second entry unusable\n");
                fseek(log_file, first_pos, SEEK_SET);
                ret = fgets(buffer, buf_size, log_file);
            } else {
                printf("**** 2nd beta size buf: '%s'\n", buffer);
                read_beta_sizes(); // skip over second entry beta sizes.
                ret = fgets(buffer, buf_size, log_file);

                printf("**** buf: '%s'\n", buffer);
                sscanf(buffer, " %ld, %ld, %e\n", &second_iter, &second_lambda_count,
                    &second_lambda_value);
                printf(": first_lambda_count: %ld\n", first_lambda_count);
                printf(": second_lambda_count: %ld\n", second_lambda_count);
                if (strncmp(buffer, "w", 1) == 0 || first_lambda_count > second_lambda_count || (first_lambda_count == second_lambda_count && first_iter > second_iter)) {
                    printf(": first entry > second_entry\n");
                    // but we can't/shouldn't use this one, go back to the first
                    fseek(log_file, first_pos, SEEK_SET);
                    ret = fgets(buffer, buf_size, log_file);
                } else {
                    printf("seeking to beginning of second entry\n");
                    fseek(log_file, second_pos, SEEK_SET);
                    ret = fgets(buffer, buf_size, log_file);
                }
            }
        } else {
            printf("log file has no second entry\n");
            fseek(log_file, first_pos, SEEK_SET);
            ret = fgets(buffer, buf_size, log_file);
        }
    }
    if (can_restore) {
        printf(": restoring from log\n");
        // buffer contains the current lambda and iter values.
        long first_iter = -1, first_lambda_count = -1;
        float first_lambda_value = -1;
        printf("final buf: '%s'\n", buffer);
        read_beta_sizes();
        ret = fgets(buffer, buf_size, log_file);

        sscanf(buffer, "%ld, %ld, %e\n", actual_iter, actual_lambda_count,
            actual_lambda_value);
        printf(": lambda_count is now %ld, lambda is now %f, iter is now %ld\n",
            *actual_lambda_count, *actual_lambda_value, *actual_iter);

        // we actually only need the beta values, which are on the current line.

        // read beta 1/2/3 from the next three lines
        auto read_beta = [&](auto beta_size, auto *beta_set) {
            long offset = 0;
            ret = fgets(buffer, buf_size, log_file);
            if (NULL == ret) {
                fprintf(stderr, "error reading log file %s\n", filename);
            }
            for (long i = 0; i < beta_size; i++) {
                // printf("values_buf: '%s'\n", buffer + offset);
                long beta_no = -1;
                float beta_val = -1.0;
                long ret = sscanf(buffer + offset, "%ld,%e, ", &beta_no, &beta_val);
                if (ret != 2) {
                    fprintf(stderr, "failed to match value in log, bad things will now happen\n");
                    fprintf(stderr, "log value was '%s'\n", buffer + offset);
                }
                // printf(": beta: %ld,%f\n", beta_no, beta_val);
                beta_set->insert_or_assign(beta_no, beta_val);
                offset += 16 + int_print_len;
            }
        };

        printf("reading beta 1\n");
        read_beta(beta1_size, &actual_beta_sets->beta1);
        printf("reading beta 2\n");
        read_beta(beta2_size, &actual_beta_sets->beta2);
        printf("reading beta 3\n");
        read_beta(beta3_size, &actual_beta_sets->beta3);
    }

    delete [] buffer;
    Rprintf("done restoring from log\n");
    return log_file;
}