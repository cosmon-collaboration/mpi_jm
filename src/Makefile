
all:
	cd sched; make  all
	cd master; make  all
	cd client; make  all
	cd demoworker; make all
	install -m 755 jm_spawnwrap ../bin
	install -m 666 startjm.py jm_machine.py ../pylib

clean:
	cd sched; make clean
	cd master; make clean
	cd client; make clean
	cd demoworker; make clean

depend:
	cd sched; make depend
	cd master; make depend
	cd client; make depend
	cd demoworker; make depend
