if [ -z $1 ]; then
    echo "must supply a directory of logs as an argument"
    exit
fi

rg -o "lasso done in [0-9]\.[0-9]* seconds" --no-filename $1 | cut -d' ' -f4