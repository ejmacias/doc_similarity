# Convert all JSON files into txt
# arg1: source dir

for filename in $1/*.json; do
    jq .text $filename --raw-output > "${filename%.*}".txt
    rm $filename
done
