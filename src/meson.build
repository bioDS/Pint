install_headers('liblasso.h')

#cc = meson.get_compiler('c')
#rt_dependency = []
#if host_machine.system() != 'windows'
#    if not cc.has_function('clock_gettime', prefix : '#include <time.h>')
#        rt_dependency = cc.find_library('rt', required: false)
#        if not cc.has_function('clock_gettime', prefix : '#include <time.h>', dependencies : rt_dependency)
#            error('clock_gettime not found')
#        endif
#    endif
#endif


liblasso_deps = [
	dependency('openmp'),
	#dependency('OpenCL'),
    dependency('glib-2.0'),
    dependency('libxxhash'),
    #dependency('boost'),
	#rt_dependency,
]

liblasso = shared_library('liblasso', 'liblasso.cpp', 'liblasso.h',
    's8b.h', 's8b.cpp',
    'queue.h', 'queue.cpp',
    'sparse_matrix.h', 'sparse_matrix.cpp',
    'regression.h', 'regression.cpp',
    'log.h', 'log.cpp',
    'csv.h', 'csv.cpp',
    'pruning.h', 'pruning.cpp',
    'tuple_val.h', 'tuple_val.cpp',
    'update_working_set.h', 'update_working_set.cpp',
	include_directories: configuration_inc,
	dependencies: liblasso_deps,
	install: true
)

libdir_abs = join_paths(get_option('prefix'), get_option('libdir'))
