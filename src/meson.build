install_headers('liblasso.h')

liblasso_deps = [
	dependency('gsl'),
	dependency('openmp'),
	dependency('glib-2.0'),
	dependency('ncurses'),
]

liblasso = shared_library('liblasso', 'liblasso.c', 'liblasso.h',
	dependencies: liblasso_deps,
	install: true
)

libdir_abs = join_paths(get_option('prefix'), get_option('libdir'))
