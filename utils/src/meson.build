install_headers('../../src/liblasso.h')

liblasso_deps = [
	dependency('gsl'),
	dependency('openmp'),
	dependency('glib-2.0'),
	dependency('ncurses'),
]

libdir_abs = join_paths(get_option('prefix'), get_option('libdir'))

lasso_exe = executable('lasso_exe', 'lasso_exe.c', link_with: liblasso,
	dependencies: liblasso_deps,
	install: true,
	install_rpath: libdir_abs
)

x2fx_exe = executable('X2_from_X', 'X2_from_X.c', link_with: liblasso, dependencies: liblasso_deps)
