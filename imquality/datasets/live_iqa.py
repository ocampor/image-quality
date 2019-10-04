import os.path

import tensorflow_datasets.public_api as tfds
import tensorflow as tf
from . import DATASETS_PATH

checksums_dir = os.path.join(DATASETS_PATH, 'url_checksums')
checksums_dir = os.path.normpath(checksums_dir)
tfds.download.add_checksums_dir(checksums_dir)


class LiveIQA(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="",  # TODO: Add description of the Live IQA dataset
            features=tfds.features.FeaturesDict({
                "distortion": tfds.features.Text(),
                "index": tf.int32,
                "distorted_image": tfds.features.Image(),
                "reference_image": tfds.features.Image(),
                "dmos": tf.float32,
                "dmos_realigned": tf.float32,
                "dmos_realigned_std": tf.float32,
            }),
            supervised_keys=("image", "mos"),
            urls=["https://dataset-homepage.org"],  # TODO: Add URL to Live IQA webpage
            citation=r"""
            @article
            """  # TODO: Add citation to article Live IQA
        )

    def _split_generators(self, manager):
        live_url = "https://data.ocampor.ai/image-quality/live.zip"
        extracted_path = manager.download_and_extract([live_url])
        images_path = os.path.join(extracted_path[0], "live")

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "images_path": images_path,
                    "labels": os.path.join(images_path, "dmos.csv")
                },
            )
        ]

    def _generate_examples(self, images_path, labels):
        with tf.io.gfile.GFile(labels) as f:
            lines = f.readlines()

        for image_id, line in enumerate(lines[1:]):
            values = line.split(",")
            yield image_id, {
                "distortion": values[0],
                "index": values[1],
                "distorted_image": os.path.join(images_path, values[2]),
                "reference_image": os.path.join(images_path, values[3]),
                "dmos": values[4],
                "dmos_realigned": values[5],
                "dmos_realigned_std": values[6],
            }
