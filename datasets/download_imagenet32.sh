#!/bin/bash
mkdir ds_imagenet
cd ds_imagenet
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-08kPTbCYHhFcwerMbZpiYFWCbbtWR17' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-08kPTbCYHhFcwerMbZpiYFWCbbtWR17" -O train_32x32.tar && rm -rf /tmp/cookies.txt
tar -xf train_32x32.tar
rm train_32x32.tar

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14CNPjwnkYAFXI77YYHSb0qa8HwIohxnh' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14CNPjwnkYAFXI77YYHSb0qa8HwIohxnh" -O valid_32x32.tar && rm -rf /tmp/cookies.txt
tar -xf valid_32x32.tar
rm valid_32x32.tar
cd ..
