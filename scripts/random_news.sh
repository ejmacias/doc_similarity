# Pick a number of random news
# arg1: source dir
# arg2: destination dir
# arg3: number of random files to move

find $1 -name "*.txt" |sort -R |head -$3 |xargs -I {} mv {} $2
