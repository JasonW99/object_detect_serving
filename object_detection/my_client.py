import tensorflow as tf
import numpy as np
from PIL import Image
# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
# Visualize the picture
from matplotlib import pyplot as plt
from utils import label_map_util
from utils import visualization_utils as vis_util


# Command line arguments
tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8)

def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Send request
    request = predict_pb2.PredictRequest()
    image = Image.open(FLAGS.image)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Call inception_v2 model to make prediction on the image
    request.model_spec.name = 'inception_v2'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['inputs'].CopyFrom(
    tf.contrib.util.make_tensor_proto(image_np_expanded))

    result = stub.Predict(request, 20.0)  # test the image with saved model, with 20 secs timeout
    # extract the useful information from the test result, and convert into nparray
    boxes = result.outputs['detection_boxes'].float_val
    classes = result.outputs['detection_classes'].float_val
    scores = result.outputs['detection_scores'].float_val
    num = result.outputs['num_detections'].float_val

    num = np.array(num).astype(np.int32)[0]
    boxes = np.array(boxes).reshape((num, 4))
    classes = np.array(classes)
    scores = np.array(scores)

    # restore the category names of the COCO dataset
    NUM_CLASSES = 90
    PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # print out the result image
    IMAGE_SIZE = (12, 8)
    vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          boxes,
          classes.astype(np.int32),
          scores,
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()
    # plt.savefig('result.png')

if __name__ == '__main__':
    tf.app.run()