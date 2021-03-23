project('lasso', 'c',
	default_options : ['c_std=gnu11'])

conf_data = configuration_data()
conf_data.set('NOT_R', true)
configure_file(output: 'config.h', configuration: conf_data)
configuration_inc = include_directories('.')

add_project_arguments([
  '-I' + meson.build_root(),
  '-ffast-math',
  '-fno-stack-protector',
  '-Ofast'
], language: 'c')

subdir('src')
subdir('tests')
subdir('utils')