AM_CFLAGS = -I /usr/lib64/glib-2.0/include/ -I /usr/include/glib-2.0/ -I /usr/include/gsl/ -I ./src/
AM_LDFLAGS = -lcurses -fopenmp -lm -L /usr/lib64/gsl/ -lgsl -L /usr/include/glib-2.0/ -lglib-2.0 -lgslcblas -L ./build/

default:
	gcc -o build/lasso_lib.o src/lasso_lib.c $(AM_CFLAGS) $(AM_LDFLAGS) -shared -fPIC
	gcc -o lasso_exe src/lasso_exe.c -L lasso_lib.o $(AM_CFLAGS) $(AM_LDFLAGS)
