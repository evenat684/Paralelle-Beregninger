ALL: all-redirect

MPI_DIR = 
# Some systems may want to add -static to the MPICC line
MPICC   = /bin/mpicc
MPIFORT = /bin/mpiff90
PNCDIR  = 
EXDIRS = ex2 ex3 ex4

CPPFLAGS = -DHAVE_CONFIG_H
CFLAGS = 
# Sample FCFLAGS for gfortran.  Note that badly named -Wall needs to be
# supplemented with *additional* warning flags (!)
#FCFLAGS = -g -Wimplicit-interface -Wimplicit-procedure -Wall -Wtabs \
#	-Wno-unused-dummy-argument
#FCFLAGS = -std=f2003 -pedantic -Wall -Wextra -Wsurprising -Wtabs \
#	-Wno-unused-dummy-argument -Wno-unused-parameter
FCFLAGS = 

MLIFE2DOBJS = mlife2d.o patch.o moptions.o mlife2d-io-stdout.o mtiming.o \
	mlife2d-pt2pt.o mlife2d-pt2ptsnd.o mlife2d-pt2ptuv.o mlife2d-fence.o \
	mlife2d-pt2pt9.o msleep.o

# .mod is also a source suffix for Modula-1 programs, so we must remove
# those rules
.SUFFIXES:
.SUFFIXES: .c .f90 .o
.f90.o:
	$(MPIFORT) $(FCFLAGS) -c $<
.c.o:
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -c $<

EXECS = mlife2d

all-redirect: $(EXECS)

msleep.mod: msleep.f90
	$(MPIFORT) $(FCFLAGS) -c msleep.f90

mlifetesting.mod mlife2d: mlife2d.mod mlife2dio.mod $(MLIFE2DOBJS)
	$(MPIFORT) $(FCFLAGS) -o mlife2d $(MLIFE2DOBJS) mlife2dmod.o

# Rules and dependencies for the modules
mlife2dmod.o mlife2d.mod: mlife2dmod.f90
	$(MPIFORT) $(FCFLAGS) -c mlife2dmod.f90
mlife2dtypes.mod mlife2dfuncs.mod: mlife2dmod.f90
	$(MPIFORT) $(FCFLAGS) -c $<
mlife2dio.mod mlife2d-io-stdout.o: mlife2d-io-stdout.f90 mlife2dtypes.mod msleep.o
	$(MPIFORT) $(FCFLAGS) -c mlife2d-io-stdout.f90

Makefile: Makefile.in ../config.status
	../config.status --file=Makefile

clean::
	rm -f *.o *.ps *.pdf $(EXECS) $(OMPEXECS) \
		fpmpi_profile.txt *.chkpt *.exe *.stackdump *.mod
	for dir in $(EXDIRS) ; do if [ -d $$dir ] ; then \
		(cd $$dir && make clean) ; fi ; done

realclean: clean
	for dir in $(EXDIRS) ; do if [ -d $$dir ] ; then \
		(cd $$dir && make realclean) ; fi ; done
	rm -f config.log config.status
	rm -rf autom4te.cache

dist: realclean
	for dir in $(EXDIRS) ; do rm $$dir/Makefile ; done
