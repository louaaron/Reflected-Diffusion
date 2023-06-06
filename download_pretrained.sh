#!/bin/bash
mkdir weights
cd weights

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AYPr0R8-3CssADBfYYSi1JuYaVrpLkTm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AYPr0R8-3CssADBfYYSi1JuYaVrpLkTm" -O cifar10.tar.gz && rm -rf /tmp/cookies.txt
tar -xvzf cifar10.tar.gz
rm cifar10.tar.gz

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CL5tM-SO4vn6tyXzrFh7VBzQv3jXDI6X' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CL5tM-SO4vn6tyXzrFh7VBzQv3jXDI6X" -O denoiser.tar.gz && rm -rf /tmp/cookies.txt
tar -xvzf denoiser.tar.gz
rm denoiser.tar.gz

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1e177im3rwI1rsHcQ5wAsaCKBKcDYRllf' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1e177im3rwI1rsHcQ5wAsaCKBKcDYRllf" -O imagenet64.tar.gz && rm -rf /tmp/cookies.txt
tar -xvzf imagenet64.tar.gz
rm imagenet64.tar.gz

cd ..