# strip notebooks
./strip_nb notebooks/1-intro.ipynb
./strip_nb notebooks/2-data-loading.ipynb

# convert notebooks to md
jupyter-nbconvert --to markdown notebooks/1-intro.ipynb
jupyter-nbconvert --to markdown notebooks/2-data-loading.ipynb
jupyter-nbconvert --to markdown notebooks/3-dpmm-v1.ipynb
jupyter-nbconvert --to markdown notebooks/4-criticism.ipynb

# count words
wc -w notebooks/1-intro.md notebooks/2-data-loading.md notebooks/3-dpmm-v1.md notebooks/4-criticism.md