set -x

##### Install Dependencies #####
apt-get update -y && apt-get upgrade -y && apt-get install zip unzip -y
pip install pyyaml pandas scikit-learn ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

##### Download Dataset #####
cd data
wget -q --show-progress -O dataset.zip "https://www.dropbox.com/scl/fi/zm7uisd9lop0j9uzok1it/CUB-200-2011.zip?rlkey=f4ip367m64lzxy4daqgsp6ilq&dl=0"
unzip -qq dataset.zip
rm -rf cvpr2016_cub
rm -rf segmentations
rm dataset.zip
