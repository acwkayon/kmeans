########################################################################
#
# Manage ML Hub Models
#
########################################################################

define MLHUB_HELP
MLHub:

  demo		Run the local demo Rscript or Python3 script.

  mllocal       Install $(APP) from cwd to ~/.mlhub.
  mltest	Install, configure, view, demo, unistall $(APP).
  mlinstall	Install the $(APP) package.
  mlconfigure	Configure the $(APP) package.
  mlreadme	View the $(APP) package README.
  mldemo	Demonstrate the $(APP) package.
  mluninstall	Uninstall the $(APP) package.


endef
export MLHUB_HELP

help::
	@echo "$$MLHUB_HELP"

########################################################################
# MODEL ARCHIVE
########################################################################

ifneq ("$(wildcard MLHUB.yaml)","")
  DESCRIPTION = MLHUB.yaml
else
  DESCRIPTION = DESCRIPTION.yaml
endif

MODEL = $(shell basename `pwd`)
MODEL_VERSION = $(shell grep version $(DESCRIPTION) | awk '{print $$NF}')
MLHUB_HOME=$(HOME)/.mlhub/$(MODEL)

# The following shell command needs to be made more robust. Works okay
# in simple MLHUB.yaml cases.

PKG=$(shell yq e '.dependencies.files' MLHUB.yaml | tr -d '-' | tr -d "\n")

README_HTML = README.html

HTML_MSG = <p>This package is part of the <a href="https://mlhub.ai">Machine Learning Hub</a> repository.</p>

.PHONY: demo
demo:
	@if [ -e demo.R ]; then Rscript demo.R; fi
	@if [ -e demo.py ]; then python3 demo.py; fi

.PHONY: mllocal
mllocal:
	if [ -n "$(PKG)" ]; then install --mode="u=rw,g=r,o=" $(PKG) $(MLHUB_HOME); fi

.PHONY: mltest mlinstall mlconfigure mlreadme mldemo mluninstall
mltest: mlinstall mlconfigure mlreadme mldemo mluninstall

mlinstall:
	@echo "-------------------------------------------------------"
	ml install gitlab:kayontoga/$(APP)

mlconfigure:
	@echo "-------------------------------------------------------"
	ml configure $(APP)

mlreadme:
	@echo "-------------------------------------------------------"
	ml readme $(APP)

mldemo: 
	@echo "-------------------------------------------------------"
	ml demo $(APP)

mluninstall:
	@echo "-------------------------------------------------------"
	ml uninstall $(APP)

.PHONY: clean
clean::
	rm -f README.txt README.html TMP.R

.PHONY: realclean
realclean::
	rm -rf __pycache__

