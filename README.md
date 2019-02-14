# IMDB Sentiment Analysis
COMP 551 [project 2](https://cs.mcgill.ca/~wlh/comp551/files/miniproject2_spec.pdf).

## Project Structure

## Installing

### Create environment

`conda create -n newenvironment --file requirements.txt`

### Download data

**Ensure your Kaggle token is in `~/.kaggle`**. You can get a new token in _My Profile->API->Create new API Token_.

`pip install kaggle --upgrade`

Download data set

`sh data_load.sh`

## Running

Once you've downloaded the data, you can reproduce the experiments done using the notebook provided.

At the end, you can make a submission with the best model found, executing `sh make_submisson.sh`


_Note: for executing the stemming, the [Punkt Sentence Tokenizer](https://www.nltk.org/_modules/nltk/tokenize/punkt.html) is necessary. It is downloaded automatically._

## Reproducibility