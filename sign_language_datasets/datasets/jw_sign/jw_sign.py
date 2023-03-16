"""Jehovah Witness Sign Language: Parallel Corpus of Sign Language Video and Text Translation based on JW Bible Verses"""
import csv
from os import path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.io.gfile import GFile

from ..warning import dataset_warning
from ...datasets.config import SignDatasetConfig
from ...utils.features import PoseFeature

_DESCRIPTION = """
Parallel Corpus of JW Sign Language, including video and translation.
Additional poses extracted by MediaPipe Holistic are also available.
"""

_CITATION = """
@inproceedings{TODO,
  title={TODO},
  author={TODO},
  booktitle={TODO},
  pages={TODO},
  year={2023}
}
@article{,
  title={TODO},
  author={TODO},
  journal={TODO},
  volume={TODO},
  pages={TODO},
  year={2023},
  publisher={TODO}
}
"""

_VIDEO_ANNOTATIONS_URL = "TODO"
_ANNOTATIONS_URL = "TODO"

_POSE_URLS = {"holistic": "TODO"}


_POSE_HEADERS = {
    "holistic": path.join(path.dirname(path.realpath(__file__)), "holistic.poseheader"),
    "openpose": path.join(path.dirname(path.realpath(__file__)), "openpose.poseheader"),
}
#_POSE_HEADERS = {"holistic": path.join(path.dirname(path.realpath(__file__)), "pose.header")}

_KNOWN_SPLITS = {
    "3.0.0-jw-verse": path.join(path.dirname(path.realpath(__file__)), "splits", "split.3.0.0-jw-verse.json")
}

def load_split(split_name: str): # TODO "-> Dict[str, List[str]]" adapt this to jw dict format
    """
    Loads a split from the file system. What is loaded must be a JSON object with the following structure:
    {"train": ..., "dev": ..., "test": ...}
    :param split_name: An identifier for a predefined split or a filepath to a custom split file.
    :return: The split loaded as a dictionary.
    """
    if split_name not in _KNOWN_SPLITS.keys():
        # assume that the supplied string is a path on the file system
        if not path.exists(split_name):
            raise ValueError(
                "Split '%s' is not a known data split identifier and does not exist as a file either.\n"
                "Known split identifiers are: %s" % (split_name, str(_KNOWN_SPLITS))
            )

        split_path = split_name
    else:
        # the supplied string is an identifier for a predefined split
        split_path = _KNOWN_SPLITS[split_name]

    with open(split_path) as infile:
        split = json.load(infile)  # type: Dict[str, List[str]]

    return split

DEFAULT_FPS = 29.970
 
  
class JWSignConfig(SignDatasetConfig):
    def __init__(self, data_type: Literal["verse"] = "verse", split: str = None, **kwargs):
        """
        :param split: An identifier for a predefined split or a filepath to a custom split file.
        :param data_type: Enforce to return verses as data.
        """
        super().__init__(**kwargs)

        self.data_type = data_type
        self.split = split

        # Verify split matches data type
        if self.split in _KNOWN_SPLITS and not self.split.endswith(self.data_type):
            raise ValueError(f"Split '{self.split}' is not compatible with data type '{self.data_type}'.")  


class JWSign(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for jw_sign dataset."""

    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {"3.0.0": "3rd release."}


    BUILDER_CONFIGS = [
        JWSignConfig(name="default", include_video=True, include_pose="holistic"),
        JWSignConfig(name="videos", include_video=True, include_pose=None),
        JWSignConfig(name="openpose", include_video=False, include_pose="openpose"),
        JWSignConfig(name="holistic", include_video=False, include_pose="holistic"),
        #JWSignConfig(name="poses", include_video=False, include_pose="holistic"),
        JWSignConfig(name="annotations", include_video=False, include_pose=None),
        JWSignConfig(name="verses", include_video=False, include_pose=None, data_type="verse"),

    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        assert isinstance(self._builder_config, JWSignConfig), \
            "Builder config for jw_sign must be an instance of JWSignConfig"
        
        features = {
            "id": tfds.features.Text(),
            "signer": tfds.features.Text(),
            "sl_id": tfds.features.Text(),
            "text": tfds.features.Text(),
        }

        if self._builder_config.include_video:
            features["fps"] = tf.int32
            features["video"] = self._builder_config.video_feature((1024, 960))

        
        if self._builder_config.include_pose is not None:
          pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
          stride = 1 if self._builder_config.fps is None else 29.970 / self._builder_config.fps

          if self._builder_config.include_pose == "openpose":
              features["pose"] = PoseFeature(shape=(None, 1, 137, 2), header_path=pose_header_path, stride=stride)
          if self._builder_config.include_pose == "holistic":
              features["pose"] = PoseFeature(shape=(None, 1, 543, 3), header_path=pose_header_path, stride=stride)
       

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="TODO",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager): #TODO def _split_generators()
        """Returns SplitGenerators."""
        dataset_warning(self)

        urls = [_VIDEO_ANNOTATIONS_URL if self._builder_config.include_video else _ANNOTATIONS_URL]

        
        if self._builder_config.include_pose != "openpose":
            for datum in index_data.values():
                del datum["openpose"]        
        
        if self._builder_config.include_pose is not None:
            urls.append(_POSE_URLS[self._builder_config.include_pose])

        downloads = dl_manager.download_and_extract(urls)
        annotations_path = path.join(downloads[0], "JWSign-release-v1", "JWSign")

        if self._builder_config.include_pose == "holistic":
            pose_path = path.join(downloads[1], "holistic")
        else:
            pose_path = None

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION, gen_kwargs={"annotations_path": annotations_path, "pose_path": pose_path, "split": "dev"},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST, gen_kwargs={"annotations_path": annotations_path, "pose_path": pose_path, "split": "test"},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN, gen_kwargs={"annotations_path": annotations_path, "pose_path": pose_path, "split": "train"},
            ),
        ]

    def _generate_examples(self, annotations_path: str, pose_path: str, split: str):
        """ Yields examples. """

        filepath = path.join(annotations_path, "annotations", "manual", "JWSign." + split + ".corpus.csv") #T0ASK 
        images_path = path.join(annotations_path, "features", "fullFrame-1024x960px", split)
        poses_path = path.join(pose_path, split) if pose_path is not None else None

        with GFile(filepath, "r") as f:
            data = csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE)
            for row in data:
                datum = {
                    "id": row["verseID"],
                    "signer": row["verse_signer"],
                    "sl_id": row["verse_lang"],
                    "text": row["verse_text"],
                }

                if self._builder_config.include_video:
                    frames_base = path.join(images_path, row["video"])[:-7] #TOASK why -7. 
                    datum["video"] = [
                        path.join(frames_base, name)
                        for name in sorted(tf.io.gfile.listdir(frames_base))
                        if name != "createDnnTrainingLabels-profile.py.lprof"
                    ]
                    datum["fps"] = self._builder_config.fps if self._builder_config.fps is not None else 29.970

                if poses_path is not None:
                    datum["pose"] = path.join(poses_path, datum["id"] + ".pose")

                yield datum["id"], datum
