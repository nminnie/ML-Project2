#!/usr/bin/env bash

$DIR=data

kaggle competitions submit -c comp-551-imbd-sentiment-classification -f $DIR/submission.csv -m "new submission"