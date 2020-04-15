import os.path

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from . import CHECKSUMS_PATH

tfds.download.add_checksums_dir(CHECKSUMS_PATH)

CITATION = r"""
@article{ponomarenko2015image,
  title={Image database TID2013: Peculiarities, results and perspectives},
  author={Ponomarenko, Nikolay and Jin, Lina and Ieremeiev, Oleg and Lukin, Vladimir and Egiazarian, Karen and Astola, Jaakko and Vozel, Benoit and Chehdi, Kacem and Carli, Marco and Battisti, Federica and others},
  journal={Signal Processing: Image Communication},
  volume={30},
  pages={57--77},
  year={2015},
  publisher={Elsevier}
}
"""
DESCRIPTION = """
The TID2013 contains 25 reference images and 3000 distorted images 
(25 reference images x 24 types of distortions x 5 levels of distortions). 
Reference images are obtained by cropping from Kodak Lossless True Color Image Suite. 
All images are saved in database in Bitmap format without any compression. File names are 
organized in such a manner that they indicate a number of the reference image, 
then a number of distortion's type, and, finally, a number of distortion's level: "iXX_YY_Z.bmp".
"""
URL = u"http://www.ponomarenko.info/tid2013.htm"
SUPERVISED_KEYS = ("distorted_image", "mos")


class Tid2013(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "distorted_image": tfds.features.Image(),
                "reference_image": tfds.features.Image(),
                "mos": tf.float32,
            }),
            supervised_keys=SUPERVISED_KEYS,
            homepage=URL,
            citation=CITATION,
        )

    def _split_generators(self, manager):
        tid2013 = "http://data.ocampor.com/tid2013.zip"
        extracted_path = manager.download_and_extract([tid2013])
        images_path = os.path.join(extracted_path[0], "tid2013")

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "images_path": images_path,
                    "labels": os.path.join(images_path, "mos.csv")
                },
            )
        ]

    def _generate_examples(self, images_path, labels):
        with tf.io.gfile.GFile(labels) as f:
            lines = f.readlines()

        for image_id, line in enumerate(lines[1:]):
            values = line.split(",")
            yield image_id, {
                "distorted_image": os.path.join(images_path, values[0]),
                "reference_image": os.path.join(images_path, values[1]),
                "mos": values[2],
            }
