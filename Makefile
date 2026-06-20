# Makefile for crystal_phonons -- 2D phonon simulator with SDL2 + SDL_ttf.
#
#   make              # release build (default target: all)
#   make debug        # -O0 -g3 with ASan + UBSan
#   make run          # build then ./crystal_phonons
#   make check-deps   # verify SDL2, SDL2_ttf, fonts, ffmpeg
#   make clean        # remove build artifacts

CC          ?= cc
CSTD        := -std=c17
WARN        := -Wall -Wextra -Wpedantic -Wshadow
OPT         := -O2
LIBS        := -lm

PKGS        := sdl2 SDL2_ttf
PKG_CFLAGS  := $(shell pkg-config --cflags $(PKGS) 2>/dev/null)
PKG_LIBS    := $(shell pkg-config --libs   $(PKGS) 2>/dev/null)

CFLAGS      ?= $(OPT) $(WARN) $(CSTD) $(PKG_CFLAGS)
LDFLAGS     ?=
LDLIBS      := $(PKG_LIBS) $(LIBS)

TARGET      := crystal_phonons
SRC         := crystal_phonons.c
OBJ         := $(SRC:.c=.o)

.PHONY: all clean run debug check-deps

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build: no optimisation, full symbols, address & UB sanitizers.
debug: CFLAGS  := -O0 -g3 $(WARN) $(CSTD) $(PKG_CFLAGS) -fsanitize=address,undefined
debug: LDLIBS  += -fsanitize=address,undefined
debug: clean $(TARGET)

run: $(TARGET)
	./$(TARGET)

check-deps:
	@echo "--- pkg-config dependencies ---"
	@for p in $(PKGS); do \
	  if pkg-config --exists $$p; then \
	    echo "OK  $$p $$(pkg-config --modversion $$p)"; \
	  else \
	    echo "MISSING  $$p"; exit 1; \
	  fi; \
	done
	@echo "--- font ---"
	@for f in /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
	          /usr/share/fonts/TTF/DejaVuSans.ttf \
	          /usr/share/fonts/dejavu/DejaVuSans.ttf; do \
	  if [ -f $$f ]; then echo "OK  $$f"; found=1; break; fi; \
	done; \
	if [ -z "$$found" ]; then \
	  echo "MISSING  DejaVuSans.ttf  (apt install fonts-dejavu fonts-dejavu-core)"; \
	  exit 1; \
	fi
	@echo "--- ffmpeg (for video capture, optional) ---"
	@if command -v ffmpeg >/dev/null 2>&1; then \
	  echo "OK  $$(ffmpeg -version | head -1)"; \
	else \
	  echo "WARNING  ffmpeg not found; video capture (V key) will not work"; \
	fi
	@echo "--- compiler ---"
	@$(CC) --version | head -1

clean:
	rm -f $(OBJ) $(TARGET)
