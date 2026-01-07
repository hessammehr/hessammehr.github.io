SHELL := /bin/bash
WORKTREE_DIR := .build
SOURCE_BRANCH := content

.PHONY: all build serve serve-only clean

all: build

$(WORKTREE_DIR)/.git:
	git worktree add --detach $(WORKTREE_DIR) $(SOURCE_BRANCH)

$(WORKTREE_DIR)/primer.css: $(WORKTREE_DIR)/.git
	cd $(WORKTREE_DIR) && \
	curl -sO https://unpkg.com/@primer/css/dist/primer.css && \
	curl -so light.css https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.7.2/styles/github.min.css && \
	curl -so dark.css https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.7.2/styles/night-owl.min.css

$(WORKTREE_DIR)/.notebooks.stamp: $(WORKTREE_DIR)/primer.css $(wildcard blog/notebooks/*.ipynb)
	cd $(WORKTREE_DIR) && \
	for file in $$(find ./blog/notebooks -name "*.ipynb" 2>/dev/null); do \
		uvx jupyter nbconvert --to markdown "$$file" --output-dir ./blog/posts; \
	done
	touch $@

$(WORKTREE_DIR)/feed.xml: $(WORKTREE_DIR)/.notebooks.stamp
	cd $(WORKTREE_DIR) && \
	cp feed.template.xml feed.xml && \
	for file in $$(find ./blog/posts -name "*.md" | sort -r); do \
		date=$$(basename "$$file" | cut -d- -f1,2,3); \
		title=$$(sed -n '1s/^# //p' "$$file"); \
		formatted_date=$$(date -j -f "%Y-%m-%d" "$$date" "+%a, %d %b %Y 00:00:00 +0000" 2>/dev/null || date -d "$$date" --rfc-822 2>/dev/null || echo "$$date"); \
		description=$$(awk 'BEGIN{p=0} /^# /{p=1;next} p==1&&NF>0{printf "%s ", $$0;exit}' "$$file" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g'); \
		sed -i.bak "s|</channel>|  <item>\n    <title>$$title</title>\n    <link>https://hessammehr.github.io/blog/posts/$$(basename $${file%.md}.html)</link>\n    <guid>https://hessammehr.github.io/blog/posts/$$(basename $${file%.md}.html)</guid>\n    <pubDate>$$formatted_date</pubDate>\n    <description>$$description</description>\n  </item>\n</channel>|" feed.xml; \
	done && rm -f feed.xml.bak

$(WORKTREE_DIR)/blog/index.md: $(WORKTREE_DIR)/.notebooks.stamp
	cd $(WORKTREE_DIR) && \
	cp blog/index.template.md blog/index.md && \
	for file in $$(find ./blog/posts -name "*.md" | sort -r); do \
		date=$$(basename "$$file" | cut -d- -f1,2,3); \
		title=$$(sed -n '1s/^# //p' "$$file"); \
		filename=$$(basename "$$file"); \
		echo "| $$date | [$$title]" >> blog/index.md; \
	done && \
	echo "" >> blog/index.md && \
	for file in $$(find ./blog/posts -name "*.md" | sort -r); do \
		title=$$(sed -n '1s/^# //p' "$$file"); \
		filename=$$(basename "$$file"); \
		echo "[$$title]: /blog/posts/$$filename" >> blog/index.md; \
	done

$(WORKTREE_DIR)/.posts.stamp: $(WORKTREE_DIR)/blog/index.md
	cd $(WORKTREE_DIR) && \
	for file in $$(find ./blog/posts -name "*.md"); do \
		pandoc -s "$$file" --template=_template.html --no-highlight -o "$${file%.md}.html"; \
	done
	touch $@

$(WORKTREE_DIR)/blog/index.html: $(WORKTREE_DIR)/.posts.stamp
	cd $(WORKTREE_DIR) && \
	pandoc -s blog/index.md -c /style.css \
		--lua-filter=<(echo 'function Link(el) el.target = el.target:gsub("%.md$$", ".html") return el end') \
		-o blog/index.html

$(WORKTREE_DIR)/CV.html: $(WORKTREE_DIR)/primer.css $(WORKTREE_DIR)/CV.md
	cd $(WORKTREE_DIR) && pandoc --section-divs -s CV.md -o CV.html

$(WORKTREE_DIR)/index.html: $(WORKTREE_DIR)/primer.css $(WORKTREE_DIR)/index.md
	cd $(WORKTREE_DIR) && pandoc -s index.md -c style.css -o index.html

build: $(WORKTREE_DIR)/feed.xml $(WORKTREE_DIR)/blog/index.html $(WORKTREE_DIR)/CV.html $(WORKTREE_DIR)/index.html

serve: build
	cd $(WORKTREE_DIR) && uv run --no-project python -m http.server 8000

clean:
	-git worktree remove $(WORKTREE_DIR) --force 2>/dev/null || rm -rf $(WORKTREE_DIR)
