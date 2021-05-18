cd center/models/
rm -rf DCNv2
git clone https://github.com/CharlesShang/DCNv2.git
cd DCNv2
./make.sh
python testcuda.py