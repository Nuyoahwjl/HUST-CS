FILE=sample

.PHONY: run

TEX = xelatex -interaction=errorstopmode
BIB = bibtex
RM = rm -fr

run: $(FILE).pdf
	cmd /c start $(FILE).pdf

$(FILE).pdf: $(FILE).tex $(FILE).aux $(FILE).toc $(FILE).bbl
	$(TEX) $(FILE)
	$(TEX) $(FILE)

$(FILE).bbl: $(FILE).bib $(FILE).aux HustGraduPaper.cls gbt7714-2005.bst gbt7714-2015.bst
	$(BIB) $(FILE)

$(FILE).aux: $(FILE).tex HustGraduPaper.cls HustBlack.eps HustGreen.eps
	$(TEX) $(FILE)

$(FILE).toc: $(FILE).aux

clean:
	$(RM) $(FILE).pdf $(FILE).toc $(FILE).log $(FILE).aux $(FILE).lof $(FILE).lot $(FILE).bbl $(FILE).blg
