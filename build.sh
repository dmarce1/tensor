mkdir $1
cd $1
cmake -DCMAKE_BUILD_TYPE=$1 ..
make -j12


