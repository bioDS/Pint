install_headers('lasso_lib.h')

lasso_lib_deps = [
	dependency('gsl'),
	dependency('openmp'),
	dependency('glib-2.0'),
]

lasso_lib = shared_library('lasso_lib', 'lasso_lib.c', 'lasso_lib.h',
	dependencies: lasso_lib_deps,
	install: true
)

libdir_abs = join_paths(get_option('prefix'), get_option('libdir'))

lasso_exe = executable('lasso_exe', 'lasso_exe.c', link_with: lasso_lib,
	dependencies: [dependency('gsl')],
	install: true,
	install_rpath: libdir_abs
)
x2fx_exe = executable('X2_from_X', 'X2_from_X.c', link_with: lasso_lib)
