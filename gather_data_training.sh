echo "Setting Up The Folders :"
mkdir plots
mkdir models

echo "Downloading Glove"
perl gdown.pl https://drive.google.com/file/d/19lKDBOWrvUqdQUUK2O5SuYMVJJQpue4R/view?usp=sharing glove.6B.100d.txt

echo "Downloading Datasets"
mkdir datasets
perl gdown.pl https://drive.google.com/file/d/14ztxuJLjHHyptAM0HVaxNio1SQcpzYfx/view?usp=sharing datasets/split-1.csv
perl gdown.pl https://drive.google.com/file/d/1j3q9MyLkbBPae7S9_dRVwTpEjEi7yN0r/view?usp=sharing datasets/split-2.csv
perl gdown.pl https://drive.google.com/file/d/1AJJmr6BqgvNk7936LxfsBLaMC-OOK1RO/view?usp=sharing datasets/split-3.csv

echo "Creating Training Data Folder And Downloading Tokenizer to the Folders"
mkdir training-data
perl gdown.pl https://drive.google.com/file/d/1x8hQlHpbcUk40T7zZenlHzSLX0OLxozU/view?usp=sharing training-data/split-1-tokenizer.pickle
perl gdown.pl https://drive.google.com/file/d/1dwlHw4v2dzHjirx9368AGxKxPnMm3JqX/view?usp=sharing training-data/split-3-tokenizer.pickle