#\!/bin/bash
# Fix formatting for a file
FILE=$1
cp $FILE ${FILE}.bak
rustfmt $FILE
