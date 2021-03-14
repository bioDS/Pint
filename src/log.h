FILE *init_log(char *filename, int n, int p, int num_betas, char **job_args,
               int job_args_num);
void save_log(int iter, double lambda_value, int lambda_count, double *beta,
              int n_betas, FILE *log_file);
int check_can_restore_from_log(char *filename, int n, int p, int num_betas,
                               char **job_args, int job_args_num);
FILE *restore_from_log(char *filename, int n, int p, int num_betas,
                       char **job_args, int job_args_num, int *actual_iter,
                       int *actual_lambda_count, double *actual_lambda_value,
                       double *actual_beta);