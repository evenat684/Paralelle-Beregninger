all: all-redirect

MPI_DIR = 
# Some systems may want to add -static to the MPICC line
MPICC   = /bin/mpicc
MPIFORT = /bin/mpiff90
PNCDIR  = 
DIRS = code-examples-c code-examples-f9x

CPPFLAGS = -DHAVE_CONFIG_H
CFLAGS = 

EXECS =

all-redirect:
	@echo "cd to code-examples-c for C, code-examples-f9x for Fortran and run make there"

Makefile: Makefile.in config.status
	./config.status

clean::
	rm -f *.o *.ps *.pdf $(EXECS) $(OMPEXECS) \
		fpmpi_profile.txt *.chkpt *.exe *.stackdump
	for dir in $(DIRS) ; do (cd $$dir && make clean) ; done

realclean: clean
	for dir in $(DIRS) ; do (cd $$dir && make realclean) ; done
	rm -f config.log config.status
	rm -rf autom4te.cache

dist: realclean
	for dir in $(DIRS) ; do rm $$dir/Makefile ; done
