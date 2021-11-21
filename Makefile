########################################################################
#
# Makefile for kmeans MLHub package
#
# Time-stamp: <Monday 2021-11-22 08:23:28 AEDT Graham Williams>
#
# Copyright (c) Graham.Williams@togaware.com
#
# License: Creative Commons Attribution-ShareAlike 4.0 International.
#
########################################################################

# App version numbers
#   Major release
#   Minor update
#   Trivial update or bug fix

APP=kmeans
VER=$(shell egrep 'version *:' MLHUB.yaml | awk '{print $$3}')
DATE=$(shell date +%Y-%m-%d)

########################################################################
# Supported Makefile modules.

# Often the support Makefiles will be in the local support folder, or
# else installed in the local user's shares.

INC_BASE=$(HOME)/.local/share/make

# Specific Makefiles will be loaded if they are found in
# INC_BASE. Sometimes the INC_BASE is shared by multiple local
# Makefiles and we want to skip specific makes. Simply define the
# appropriate INC to a non-existant location and it will be skipped.

INC_MODULE=$(INC_BASE)/modules.mk

ifneq ("$(wildcard $(INC_MODULE))","")
  include $(INC_MODULE)
endif

########################################################################
# HELP
#
# Help for targets defined in this Makefile.

define HELP
$(APP):

  test	Run a suite of tests.
  demo	Run the demo locally.	

endef
export HELP

help::
	@echo "$$HELP"

########################################################################
# LOCAL TARGETS

test:
	python train.py 3 --input iris.csv
	python train.py 3 --input=iris.csv
	python train.py 3 -i iris.csv
	python train.py 3 < iris.csv
	cat iris.csv | python train.py 3                                                                                                         
	echo "" | python train.py 3

