all: moments.h moments.c distances.h distances.c auxiliary.h auxiliary.c
	$(CC) -shared -o libmoments.so -O3 -fPIC -g moments.c
	$(CC) -shared -o libdistances.so -O3 -fPIC -g distances.c
	$(CC) -shared -o libauxiliary.so -O3 -fPIC -g auxiliary.c

