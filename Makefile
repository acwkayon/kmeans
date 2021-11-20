test:
	python train.py 3 --input iris.csv
	python train.py 3 --input=iris.csv
	python train.py 3 -i iris.csv
	python train.py 3 < iris.csv
	cat iris.csv | python train.py 3                                                                                                         
	echo "" | python train.py 3

demo:
	python demo.py
