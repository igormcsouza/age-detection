docker run --rm -it \
-v `pwd`/scripts:/age-detection \
-v `pwd`/data:/data \
--device=/dev/video0:/dev/video0 \
igormcsouza/ml:age-detection \
python3 detect-age.py -i /$@ -f face-detector -a age-detector