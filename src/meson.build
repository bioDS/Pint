install_headers('liblasso.h')

cc = meson.get_compiler('c')
rt_dependency = []
if host_machine.system() != 'windows'
    if not cc.has_function('clock_gettime', prefix : '#include <time.h>')
        rt_dependency = cc.find_library('rt', required: false)
        if not cc.has_function('clock_gettime', prefix : '#include <time.h>', dependencies : rt_dependency)
            error('clock_gettime not found')
        endif
    endif
endif

liblasso_deps = [
	dependency('gsl'),
	dependency('openmp'),
	dependency('glib-2.0'),
	dependency('ncurses'),
	rt_dependency,
]

liblasso = shared_library('liblasso', 'liblasso.c', 'liblasso.h',
	dependencies: liblasso_deps,
	install: true
)

libdir_abs = join_paths(get_option('prefix'), get_option('libdir'))
