PYTHON ?= python3
VERSION ?=

.PHONY: help stale check publish

help:
	@echo "MIWN — make targets"
	@echo ""
	@echo "  make stale                  flag solutions built against a stale methods pin"
	@echo "  make check                  stale --strict (nonzero exit if anything stale) — CI gate"
	@echo "  make publish VERSION=<lbl>  assemble a release snapshot for <lbl>"
	@echo ""
	@echo "Methods pin: the standards/ submodule. Each solution records the"
	@echo "standards_methods_sha it was built with in its meta.json; stale.py"
	@echo "compares those against the current submodule pin."

stale:
	@$(PYTHON) scripts/stale.py

check:
	@$(PYTHON) scripts/stale.py --strict

publish:
	@test -n "$(VERSION)" || { echo "ERROR: set VERSION=<label>, e.g. make publish VERSION=ecta-accepted"; exit 2; }
	@$(PYTHON) scripts/publish.py --version "$(VERSION)"
