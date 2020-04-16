install_headers('lasso_lib.h')

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

lasso_lib_deps = [
	dependency('gsl'),
	dependency('openmp'),
	dependency('glib-2.0'),
	dependency('ncurses'),
	rt_dependency,
]

lasso_lib = shared_library('lasso_lib', 'lasso_lib.c', 'lasso_lib.h',
	dependencies: lasso_lib_deps,
	install: true
)

libdir_abs = join_paths(get_option('prefix'), get_option('libdir'))

lasso_exe = executable('lasso_exe', 'lasso_exe.c', link_with: lasso_lib,
	dependencies: lasso_lib_deps,
	install: true,
	install_rpath: libdir_abs
)

x2fx_exe = executable('X2_from_X', 'X2_from_X.c', link_with: lasso_lib, dependencies: lasso_lib_deps)
