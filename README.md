# Announcement

- May, 7, 2024: ⚠️ This repository is still incomplete, so it has many bugs. It's planned to be continuously updated until the final deadline of YAICON, with the goal of having it functioning properly by then.

# Installation

## Installation via GitHub
```sh
git clone {{this-repo}}
cd yaicon-tight-report-review
conda env create -f environment.yaml
```

## Tutorials

### 1. Pull the external datas
You need to pull the external datas (reports, papers, ...) from external API (ex. Notion, arXiv) to `yaicon-tight-report-review/reports` before launch the program.

### 2. Launch the program
`--keywords`: Google search keywords for academic papers

```sh
python src/main.py --keywords "instructblip"
```
>The `--keywords` arguments are editable for paper title, theme, categories and etc ...
\
**ex) --keywords "clip" "instructblip" "unet" "vit"**

### 3. Checkout the results
```sh
cd results
```