install_headers('liblasso.h')
install_headers('/home/kieran/work/lasso_testing/TurboPFor/vp4.h')

conf_data = configuration_data()
conf_data.set('NOT_R', true)
configure_file(output: 'config.h', configuration: conf_data)
configuration_inc = include_directories('.')

cc = meson.get_compiler('c')

vp4c = cc.find_library('vp4c', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])
vp4d = cc.find_library('vp4d', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])
bitpack = cc.find_library('bitpack', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])
bitunpack = cc.find_library('bitunpack', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])
bitutil = cc.find_library('bitutil', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])
vint = cc.find_library('vint', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])
bitunpack_sse = cc.find_library('bitunpack_sse', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])

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
	#dependency('glib-2.0'),
	dependency('ncurses'),
	vp4c,
	vp4d,
	bitpack,
	bitunpack,
	bitunpack_sse,
	bitutil,
	vint,
	rt_dependency,
]

liblasso = shared_library('liblasso', 'liblasso.c', 'liblasso.h',
	include_directories: configuration_inc,
	dependencies: liblasso_deps,
	install: true
)

libdir_abs = join_paths(get_option('prefix'), get_option('libdir'))
