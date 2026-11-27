TECTONIC ?= $(shell command -v tectonic 2>/dev/null || echo /home/user/bot/.bin/tectonic)
PYTHON   ?= $(shell [ -x /home/user/bot/.paper_venv/bin/python ] && echo /home/user/bot/.paper_venv/bin/python || echo python3)

.PHONY: all banner tables figures arch pdf preview clean

all: banner arch tables figures pdf

banner:
	$(PYTHON) ../scripts/paper/make_banner.py

arch: figures/architecture.pdf

figures/architecture.pdf: figures/architecture.tex
	cd figures && $(TECTONIC) architecture.tex

tables:
	$(PYTHON) ../scripts/paper/make_tables.py

figures:
	$(PYTHON) ../scripts/paper/make_figures.py

pdf: main.pdf

main.pdf: main.tex refs.bib figures/architecture.pdf $(wildcard sections/*.tex) $(wildcard tables/*.tex) $(wildcard figures/*.pdf)
	$(TECTONIC) main.tex

preview: main.pdf
	mkdir -p build_pages
	pdftoppm -png -r 150 main.pdf build_pages/page

clean:
	rm -rf main.pdf main.aux main.bbl main.blg main.log main.out build_pages
