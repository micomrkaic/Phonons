# Makefile for crystal_phonons -- 2D lattice-dynamics playground (SDL2 + SDL_ttf).
#
#   make              # release build (default target: all)
#   make debug        # -O0 -g3 with ASan + UBSan
#   make run          # build then ./crystal_phonons
#   make selftest     # build then run the numerical validation suite
#   make install      # binary -> ~/.local/bin, man page -> ~/.local/share/man/man1
#   make uninstall    # remove both
#   make wasm         # browser build with Emscripten (needs emcc on PATH)
#   make check-deps   # verify SDL2, SDL2_ttf, fonts, ffmpeg
#   make clean        # remove build artifacts
#
# Install locations follow the XDG user layout.  Override as needed:
#   make install PREFIX=/usr/local        (system-wide; needs sudo)
#   make install DESTDIR=/tmp/stage       (staged install for packaging)

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
MAN         := crystal_phonons.1

PREFIX      ?= $(HOME)/.local
BINDIR      ?= $(PREFIX)/bin
MANDIR      ?= $(PREFIX)/share/man/man1
DESTDIR     ?=
INSTALL     ?= install

EMCC        ?= emcc
WASM_DIR    := build/wasm
FONT_DIR    ?= /usr/share/fonts/truetype/dejavu

.PHONY: all clean run debug selftest install uninstall wasm check-deps

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

selftest: $(TARGET)
	./$(TARGET) --selftest

install: $(TARGET)
	$(INSTALL) -d "$(DESTDIR)$(BINDIR)" "$(DESTDIR)$(MANDIR)"
	$(INSTALL) -m 755 $(TARGET) "$(DESTDIR)$(BINDIR)/$(TARGET)"
	$(INSTALL) -m 644 $(MAN) "$(DESTDIR)$(MANDIR)/$(MAN)"
	@echo "installed $(DESTDIR)$(BINDIR)/$(TARGET)"
	@echo "installed $(DESTDIR)$(MANDIR)/$(MAN)"
	@case ":$$PATH:" in \
	  *":$(BINDIR):"*) ;; \
	  *) echo "note: $(BINDIR) is not on your PATH; add it in your shell rc";; \
	esac

uninstall:
	rm -f "$(DESTDIR)$(BINDIR)/$(TARGET)" "$(DESTDIR)$(MANDIR)/$(MAN)"
	@echo "removed $(TARGET) and $(MAN) from $(DESTDIR)$(PREFIX)"

# Browser build.  Fonts are embedded at /fonts inside the virtual FS;
# video capture and screenshots are compiled out.
wasm:
	@command -v $(EMCC) >/dev/null 2>&1 || { echo "emcc not found (install emsdk)"; exit 1; }
	mkdir -p $(WASM_DIR)/fonts
	cp $(FONT_DIR)/DejaVuSans.ttf $(FONT_DIR)/DejaVuSansMono.ttf $(WASM_DIR)/fonts/
	$(EMCC) -O2 $(CSTD) -sUSE_SDL=2 -sUSE_SDL_TTF=2 -sALLOW_MEMORY_GROWTH=1 \
	    --embed-file $(WASM_DIR)/fonts@/fonts \
	    -o $(WASM_DIR)/$(TARGET).html $(SRC)
	@echo "open $(WASM_DIR)/$(TARGET).html via a local web server, e.g."
	@echo "    python3 -m http.server -d $(WASM_DIR)"

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
	          /usr/share/fonts/dejavu/DejaVuSans.ttf \
	          /usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf \
	          /System/Library/Fonts/Supplemental/Arial.ttf; do \
	  if [ -f $$f ]; then echo "OK  $$f"; found=1; break; fi; \
	done; \
	if [ -z "$$found" ]; then \
	  echo "MISSING  DejaVuSans.ttf  (apt install fonts-dejavu, or set PHONONS_FONT)"; \
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
	rm -rf build
