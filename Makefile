test:
	ml train kmeans 3 --input test.csv
	ml train kmeans 3 --input=test.csv
	ml train kmeans 3 -i test.csv
	ml train kmeans 3 < test.csv
	cat test.csv | ml train kmeans 3                                                                                                         
	echo "" | ml train kmeans 3
