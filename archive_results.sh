#!/bin/bash
rm -Rf *~
tar -pczf BLBN.Results.`date +"%Y-%m-%d"`.tar.gz BLBN/results/
