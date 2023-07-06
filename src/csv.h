#ifndef csv_h
#define csv_h

XMatrix read_x_csv(const char* fn, int_fast64_t n, int_fast64_t p);
float*  read_y_csv(const char* fn, int_fast64_t n);

#endif
