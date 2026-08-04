# Builds the reference driver against the pinned upstream catch22 C sources.
#
# Two variants so the Rust-vs-C comparison cannot be accused of hobbling the
# baseline:
#   driver_O2       -O2, portable
#   driver_native   -O3 -march=native
#
# Run `bash fetch.sh` first to populate upstream/.

UPSTREAM := upstream/C
SRC := $(filter-out $(UPSTREAM)/main.c,$(wildcard $(UPSTREAM)/*.c))
# -std=c99 matches how upstream builds these sources (see their setup.py); the
# POSIX define is needed for getline/clock_gettime in our driver, which c99
# alone would hide.
CFLAGS_COMMON := -std=c99 -D_POSIX_C_SOURCE=200809L -I$(UPSTREAM) \
                 -Wno-unused-result -Wno-alloc-size-larger-than
LDLIBS := -lm

.PHONY: all clean
all: driver_O2 driver_native

driver_O2: driver.c $(SRC)
	$(CC) $(CFLAGS_COMMON) -O2 -o $@ $^ $(LDLIBS)

driver_native: driver.c $(SRC)
	$(CC) $(CFLAGS_COMMON) -O3 -march=native -o $@ $^ $(LDLIBS)

clean:
	rm -f driver_O2 driver_native
