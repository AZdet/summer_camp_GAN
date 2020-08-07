if [ ! -f  "decoder_29.pt" ]; then
  wget --no-check-certificate 'https://www.dropbox.com/s/yt8gsdx3ppenaaq/decoder_29.pt?dl=0' -O models/decoder_29.pt
fi
URL=https://www.dropbox.com/s/zdq6roqf63m0v5f/celeba-256x256-5attrs.zip?dl=0
ZIP_FILE=./stargan_celeba_256/models/celeba-256x256-5attrs.zip
mkdir -p ./stargan_celeba_256/models/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./stargan_celeba_256/models/