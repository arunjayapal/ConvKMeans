# This is a Python template Makefile, do modification as you want
#
# Project: 
# Author:
# Email :

HOST = 127.0.0.1
PYTHONPATH="$(shell printenv PYTHONPATH):$(PWD)"

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force  {} +

run:

test:
	PYTHONPATH=$(PYTHONPATH) python 

test-kmeans-optimizer:
	PYTHONPATH=$(PYTHONPATH) python ./convk/scripts/test_kmeans_optimizer.py

test-kmeans-pytorch:
	PYTHONPATH=$(PYTHONPATH) python ./convk/scripts/test_kmeans_pytorch.py

cleanall:
