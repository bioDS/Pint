


#TurboPFor/libvp4d.a:
#	STATIC=1 CFLAGS=-fPIC make -C TurboPFor libvp4d.a libvp4c.a libbitunpack.a libbitpack.a libbitutil.a libvint.a libbitunpack_sse.a

all:
	STATIC=1 make -C TurboPFor

clean:
	find . -type f -name libseidel.o -delete -or -name test_seidel -delete
	make -C TurboPFor clean

