SHELL := /bin/bash

.PHONY: all build serve clean

OUT_DIR := .build
NBCONVERT := uvx --from jupyter-core --with nbconvert jupyter nbconvert

NOTEBOOKS := $(wildcard blog/notebooks/*.ipynb)
NOTEBOOK_MDS := $(patsubst blog/notebooks/%.ipynb,$(OUT_DIR)/blog/posts/%.md,$(NOTEBOOKS))
MDS := $(patsubst blog/posts/%.md,$(OUT_DIR)/blog/posts/%.md,$(wildcard blog/posts/*.md))
ALL_MDS := $(MDS) $(NOTEBOOK_MDS)
ALL_HTMLS := $(patsubst %.md,%.html,$(ALL_MDS))

IMAGES := $(patsubst blog/images/%,$(OUT_DIR)/blog/images/%,$(wildcard blog/images/*))

all: build

$(OUT_DIR)/primer.css: | $(OUT_DIR)
	curl -so $(OUT_DIR)/primer.css https://unpkg.com/@primer/css/dist/primer.css

$(OUT_DIR)/light.css: | $(OUT_DIR)
	curl -so $(OUT_DIR)/light.css https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.7.2/styles/github.min.css

$(OUT_DIR)/dark.css: | $(OUT_DIR)
	curl -so $(OUT_DIR)/dark.css https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.7.2/styles/github-dark.min.css

# Plain .md posts
$(OUT_DIR)/blog/posts/%.md: blog/posts/%.md | $(OUT_DIR)/blog/posts
	cp "$<" $(OUT_DIR)/blog/posts/

# .md posts from .ipynb
$(OUT_DIR)/blog/posts/%.md: blog/notebooks/%.ipynb | $(OUT_DIR)/blog/posts
	$(NBCONVERT) --to markdown "$<" --output-dir $(OUT_DIR)/blog/posts

# .html posts from .md
$(OUT_DIR)/blog/posts/%.html: $(OUT_DIR)/blog/posts/%.md $(OUT_DIR)/primer.css $(OUT_DIR)/light.css $(OUT_DIR)/dark.css
	pandoc -s "$<" --template=_template.html --syntax-highlighting=none --mathjax -o "$@"

$(OUT_DIR)/feed.xml: $(ALL_MDS) scripts/generate_feed.py
	uv run --no-project python scripts/generate_feed.py $(ALL_MDS) > $@

$(OUT_DIR)/blog/index.md: $(ALL_MDS) $(ALL_HTMLS) $(IMAGES)
	cp blog/index.template.md $(OUT_DIR)/blog/index.md && \
	for file in $$(find $(OUT_DIR)/blog/posts -name "*.md" | sort -r); do \
		date=$$(basename "$$file" | cut -d- -f1,2,3); \
		title=$$(sed -n '1s/^# //p' "$$file"); \
		filename=$$(basename "$$file"); \
		echo "| $$date | [$$title]" >> $(OUT_DIR)/blog/index.md; \
	done && \
	echo "" >> $(OUT_DIR)/blog/index.md && \
	for file in $$(find $(OUT_DIR)/blog/posts -name "*.md" | sort -r); do \
		title=$$(sed -n '1s/^# //p' "$$file"); \
		filename=$$(basename "$$file"); \
		echo "[$$title]: /blog/posts/$$filename" >> $(OUT_DIR)/blog/index.md; \
	done

$(OUT_DIR)/blog/index.html: $(OUT_DIR)/blog/index.md
	cd $(OUT_DIR) && \
	pandoc -s blog/index.md -c /style.css \
		--lua-filter=<(echo 'function Link(el) el.target = el.target:gsub("%.md$$", ".html") return el end') \
		-o blog/index.html

$(OUT_DIR)/CV.html: $(OUT_DIR)/primer.css CV.md
	pandoc --section-divs -s CV.md -o $(OUT_DIR)/CV.html

$(OUT_DIR)/%: % | $(OUT_DIR)
	@mkdir -p $(dir $@)
	cp $< $@

$(OUT_DIR)/index.html: index.md $(OUT_DIR)/primer.css $(OUT_DIR)/style.css $(OUT_DIR)/rings.png
	pandoc -s $< -c style.css -o $@

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
