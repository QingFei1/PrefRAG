set -e
set -x

pip install gdown
mkdir -p .temp/
# URL: https://drive.google.com/file/d/1eScMzopr07qMk2Q4sG7dT8l6bprTxLBG/view?usp=sharing
gdown "1eScMzopr07qMk2Q4sG7dT8l6bprTxLBG&confirm=t" -O .temp/data.zip
unzip  -o .temp/data.zip -x "*.DS_Store" "__MACOSX/*"

rm -rf .temp/
