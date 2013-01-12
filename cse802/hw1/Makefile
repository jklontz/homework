all: homework

homework: hardware
	pdflatex homework.tex
	pdflatex homework.tex

hardware:
	Rscript hardware.R

clean:
	rm -f *.aux *.log *.pdf *.synctex.gz
