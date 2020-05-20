import argparse
import json
import math
import os

import imgaug
import keras_ocr
from sklearn.model_selection import train_test_split
import tensorflow as tf


def model(dataset: list, model_dir: str) -> keras_ocr.detection.Detector:
    """Load a detector model and train on dataset, saving checkpoints and 
    resultant model to model_dir, which can/ought to be an S3 path

    Arguments:
        dataset {list} -- list of samples

    Keyword Arguments:
        model_dir {str} -- where to save checkpoints and model

    Returns:
        keras_ocr.detection.Detector -- instance of a detector model
    """    

    train, validation = train_test_split(
        dataset, train_size=0.8, random_state=42
    )

    augmenter = imgaug.augmenters.Sequential([
        imgaug.augmenters.Affine(
        scale=(1.0, 1.2),
        rotate=(-5, 5)
        ),
        imgaug.augmenters.GaussianBlur(sigma=(0, 0.5)),
        imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2)
    ])

    generator_kwargs = {'width': 640, 'height': 640}
    
    training_image_generator = keras_ocr.datasets.get_detector_image_generator(
        labels=train,
        **generator_kwargs
    )
    
    validation_image_generator = keras_ocr.datasets.get_detector_image_generator(
        labels=validation,
        **generator_kwargs
    )
    
    detector = keras_ocr.detection.Detector()
    
    batch_size = 1
    
    training_generator, validation_generator = [
        detector.get_batch_generator(
            image_generator=image_generator, batch_size=batch_size
        ) for image_generator in
        [training_image_generator, validation_image_generator]
    ]
    
    detector.model.fit_generator(
        generator=training_generator,
        steps_per_epoch=math.ceil(len(train) / batch_size),
        epochs=1,
        workers=0, # execute the generator on the main thread
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                restore_best_weights=True, 
                patience=5
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, 'detector_icdar2013.h5')
            )
        ],
        validation_data=validation_generator,
        validation_steps=math.ceil(len(validation) / batch_size)
    )

    # TODO: add model.evaluate()

    return detector


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    # Model arguments
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=640)

    # TODO: add augment options
    # TODO: add input size for generator_kwargs

    return parser.parse_known_args()


if __name__ == "__main__":
    args, _ = _parse_args()

    model_dir = args.sm_model_dir

    dataset = keras_ocr.datasets.get_icdar_2013_detector_dataset(
        cache_dir='.',
        skip_illegible=True
    )
    
    detector = model(dataset, model_dir=model_dir)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '001'
        detector.model.save(
            os.path.join(model_dir, '001'),
            'detecotr_model.h5'
        )
