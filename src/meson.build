install_headers('liblasso.h')

conf_data = configuration_data()
conf_data.set('NOT_R', true)
configure_file(output: 'config.h', configuration: conf_data)
configuration_inc = include_directories('.')

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
	rt_dependency,
]

liblasso = shared_library('liblasso', 'liblasso.c', 'liblasso.h',
    's8b.h', 's8b.c',
    'queue.h', 'queue.c',
    'sparse_matrix.h', 'sparse_matrix.c',
    'regression.h', 'regression.c',
    'log.h', 'log.c',
    'csv.h', 'csv.c',
    'pruning.h', 'pruning.c',
	include_directories: configuration_inc,
	dependencies: liblasso_deps,
	install: true
)

libdir_abs = join_paths(get_option('prefix'), get_option('libdir'))
