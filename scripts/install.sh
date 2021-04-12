# create venv with all required dependencies
conda create --name grapher -y

# activate venv
source activate grapher

# install cuda
# (uncomment below if this is your CUDA version, otherwise change 9.0 with your CUDA version)
# conda install cudatoolkit==9.0 -c anaconda -y
conda install pip -y

# install pytorch
conda install pytorch -c pytorch -y

# install packages
conda install numpy scipy pandas scikit-learn networkx docopt pyyaml matplotlib requests seaborn pyemd ipython -y

# compile ORCA
g++ -O2 -std=c++11 -o utils/orca/orca.exe utils/orca/orca.cpp