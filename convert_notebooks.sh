#!/bin/bash

# Create a folder for converted notebooks
mkdir -p _posts

# Loop through each notebook in the _notebooks folder
for notebook in _notebooks/*.ipynb
do
  # Get the base name of the notebook
  base=$(basename "$notebook" .ipynb)
  
  # Convert the notebook to markdown
  jupyter nbconvert --to markdown "$notebook"
  
  # Move the converted notebook to the _posts folder
  mv "${base}.md" "_posts/${base}.md"
  
  # Move associated files (images, etc.) to the appropriate folder
  if [ -d "${base}_files" ]; then
    mv "${base}_files" "_posts/"
  fi
done
