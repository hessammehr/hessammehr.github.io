SHELL := /bin/bash

.PHONY: all build serve clean

OUT_DIR := .build
NBCONVERT := uvx --from jupyter-core --with nbconvert jupyter nbconvert

NOTEBOOKS := $(wildcard blog/notebooks/*.ipynb)
NOTEBOOK_MDS := $(patsubst blog/notebooks/%.ipynb,$(OUT_DIR)/blog/posts/%.md,$(NOTEBOOKS))
MDS := $(patsubst blog/posts/%.md,$(OUT_DIR)/blog/posts/%.md,$(wildcard blog/posts/*.md))
HTMLS := $(patsubst blog/htmls/%.html,$(OUT_DIR)/blog/posts/%.html,$(wildcard blog/htmls/*.html))
ALL_MDS := $(MDS) $(NOTEBOOK_MDS)
ALL_HTMLS := $(patsubst %.md,%.html,$(ALL_MDS)) $(HTMLS)

IMAGES := $(patsubst blog/images/%,$(OUT_DIR)/blog/images/%,$(wildcard blog/images/*))

all: build

$(OUT_DIR)/primer.css: | $(OUT_DIR)
	curl -sfo $(OUT_DIR)/primer.css https://unpkg.com/@primer/css/dist/primer.css

$(OUT_DIR)/light.css: | $(OUT_DIR)
	curl -sfo $(OUT_DIR)/light.css https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.5.0/styles/github.min.css

$(OUT_DIR)/dark.css: | $(OUT_DIR)
	curl -sfo $(OUT_DIR)/dark.css https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.5.0/styles/github-dark.min.css

$(OUT_DIR)/highlight.min.js: | $(OUT_DIR)
	curl -sfo $(OUT_DIR)/highlight.min.js https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.5.0/highlight.min.js

# Plain .md posts
$(OUT_DIR)/blog/posts/%.md: blog/posts/%.md | $(OUT_DIR)/blog/posts
	cp "$<" $(OUT_DIR)/blog/posts/

# HTML posts
$(OUT_DIR)/blog/posts/%.html: blog/htmls/%.html | $(OUT_DIR)/blog/posts
	cp "$<" $(OUT_DIR)/blog/posts/

# .md posts from .ipynb
$(OUT_DIR)/blog/posts/%.md: blog/notebooks/%.ipynb | $(OUT_DIR)/blog/posts
	$(NBCONVERT) --to markdown "$<" --output-dir $(OUT_DIR)/blog/posts

# .html posts from .md
$(OUT_DIR)/blog/posts/%.html: $(OUT_DIR)/blog/posts/%.md $(OUT_DIR)/primer.css $(OUT_DIR)/light.css $(OUT_DIR)/dark.css $(OUT_DIR)/highlight.min.js
	title=$$(sed -n '1s/^# \(Draft: \)\{0,1\}//p' "$<"); \
	sed '1s/^# Draft: /# [Draft]{.draft-title} /' "$<" | \
	pandoc -s --template=_template.html --syntax-highlighting=none --mathjax --metadata=pagetitle="$$title" -o "$@"

$(OUT_DIR)/feed.xml: $(ALL_MDS) scripts/generate_feed.py
	uv run --no-project python scripts/generate_feed.py $(ALL_MDS) > $@

$(OUT_DIR)/blog/index.md: $(ALL_HTMLS) $(IMAGES)
	cp blog/index.template.md $@ && \
	for file in $$(echo "$(ALL_HTMLS)" | tr ' ' '\n' | sort -r); do \
		date=$$(basename "$$file" | cut -d- -f1,2,3); \
		mdfile="$${file%.html}.md"; \
		if [ -f "$$mdfile" ]; then \
			title=$$(sed -n '1s/^# //p' "$$mdfile"); \
		else \
			title=$$(sed -n 's/.*<title>\(.*\)<\/title>.*/\1/p' "$$file" | head -1); \
		fi; \
		filename=$$(basename "$$file"); \
		if echo "$$title" | grep -q '^Draft: '; then \
			title=$$(echo "$$title" | sed 's/^Draft: //'); \
			echo "| $$date | <span class=\"draft-label\">Draft</span>[$$title](/blog/posts/$$filename)" >> $@; \
		else \
			echo "| $$date | [$$title](/blog/posts/$$filename)" >> $@; \
		fi; \
	done

$(OUT_DIR)/blog/index.html: $(OUT_DIR)/blog/index.md
	pandoc -s $< -c /style.css -o $@

$(OUT_DIR)/CV.html: $(OUT_DIR)/CV.css CV.md 
	pandoc --section-divs -s CV.md -c CV.css -o $(OUT_DIR)/CV.html

$(OUT_DIR)/%: % | $(OUT_DIR)
	@mkdir -p $(dir $@)
	cp $< $@

$(OUT_DIR)/index.html: index.md $(OUT_DIR)/primer.css $(OUT_DIR)/style.css $(OUT_DIR)/rings.png $(OUT_DIR)/lines.svg $(OUT_DIR)/lampshade.jpeg
	pandoc -s $< -c style.css --metadata=pagetitle="Hessam Mehr" -o $@

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/blog/posts:
	mkdir -p $(OUT_DIR)/blog/posts

build: $(OUT_DIR)/feed.xml $(OUT_DIR)/blog/index.html $(OUT_DIR)/CV.html $(OUT_DIR)/index.html

watch: 
	@while true; do \
		make build; \
		sleep 5; \
	done

serve: build
	uv run --no-project python -m http.server 8000 -d $(OUT_DIR)

clean:
	rm -rf $(OUT_DIR)

test:
	@echo "NOTEBOOKS: $(NOTEBOOKS)"
	@echo "NOTEBOOK_MDS: $(NOTEBOOK_MDS)"
	@echo "MDS: $(MDS)"
	@echo "ALL_MDS: $(ALL_MDS)"
	@echo "ALL_HTMLS: $(ALL_HTMLS)"
	@echo "IMAGES: $(IMAGES)"
