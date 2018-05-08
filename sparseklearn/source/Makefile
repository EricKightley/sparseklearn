all: moments.h moments.c distances.h distances.c
	$(CC) -shared -o libmoments.so -O3 -fPIC -g moments.c
	$(CC) -shared -o libdistances.so -O3 -fPIC -g distances.c

