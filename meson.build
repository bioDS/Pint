project('lasso', 'cpp',
	default_options : ['b_lto=true'])

configuration_inc = include_directories('.')
conf_data = configuration_data()
conf_data.set('NOT_R', true)
configure_file(output: 'config.h', configuration: conf_data)

add_project_arguments([
  '-ffast-math',
  '-fno-stack-protector',
  '-fpermissive',
  '-Wno-unused',
 #'-Ofast',
 # '-g',
  # '-Os',
  # '-fno-inline',
], language: 'cpp')

subdir('src')
subdir('tests')
subdir('utils')
