SHELL:=/bin/bash
URLgit=https://github.com/fleaf5/rm_tool_for_raw_data.git

.PHONY: gadd
gadd:
	@read -p "git commit description: " DESCRIPTION;\
	git add . ;\
	git commit -m "$$DESCRIPTION" 

.PHONY: gpush
gpush:
	git push origin master

gpull:
	git pull origin master --allow-unrelated-histories

gcheck:
	git log -p -2

gremote:
	git remote add -m github $$URLgit

.PHONY: clean
clean: 
	rm */*.pyc
	rm */*.bak
