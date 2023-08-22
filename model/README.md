# otx train setting

## 
```
cd ./workspace

otx find --task detection

otx build --train-data-root ./(cocodataset directory) --model (model name) --workspace new_directory(directory name)
ass
cd new_directory

ds_count splitted_dataset/ 2

(parameter 수정)
otx train params --learning_parameters.batch_size 64 --learning_parameters.num_iters 1
gedit template.yaml
gedit configuration.yaml

otx train

otx eval --test-data-roots ./splitted_dataset/val

otx export

find -name "modelname"

otx deploy ./workspace/.otx/lib/python3.10/site-packages/otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yalm(find -name "modelname"결과) --load-weights outputs/20230818_13856_export/openvino/openvino.xml(생성된 xml)

cd outputs/20230818_13856_deploy(생성된 deploy directory)

unzip openvino.zip

cd python/

pip install -r requirements.txt

python demo.py --input (video or image) --models ../model
```
