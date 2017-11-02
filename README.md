### 0. download the pretrained model
```bash
cd objectDetectServing/object_detection
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz
tar xvzf ssd_inception_v2_coco_11_06_2017.tar.gz
rm ssd_inception_v2_coco_11_06_2017.tar.gz
```

### 1. setup environment
```bash
cd ..
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

### 2. build pretrained model to server
```bash
python object_detection/my_export_inference_graph.py \
--input_type=image_tensor \
--pipeline_config_path=object_detection/samples/configs/ssd_inception_v2_coco.config \
--trained_checkpoint_prefix=object_detection/ssd_inception_v2_coco_11_06_2017/model.ckpt \
--output_directory /tmp/inception_v2_model/ \
--model_version 1
```

### 3. setup server
change the directory to the folder of tensorflow_serving 
```bash
cd ~
cd serving
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=inception_v2 --model_base_path=/tmp/inception_v2_model/ --enable_batching=true
```

### 4. test 
open another terminal
```bash
cd objectDetectServing

protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python object_detection/my_client.py --server=localhost:9000 --image=object_detection/test_images/image1.jpg	
```
### 5. notes
the directories "object_detection" and "slim" are from the tensorflow repository "tensorflow/models/research/".</ br>
the files "my_client.py", "my_export_inference_graph.py", and "my_exporter" are the original codes or modified codes.




