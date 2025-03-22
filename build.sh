mkdir $1
mkdir $1/tensor/
cd $1
cmake -DCMAKE_BUILD_TYPE=$1 ..
make -j12


