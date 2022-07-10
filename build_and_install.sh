cd pointnet2_lib/pointnet2
rm -r build
python setup.py install
cd ../../

cd lib/utils/iou3d/
rm -r build
python setup.py install

cd ../roipool3d/
rm -r build
python setup.py install

cd ../../../tools
