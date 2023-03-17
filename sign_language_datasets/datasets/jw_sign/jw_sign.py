"""Jehovah Witness Sign Language: Parallel Corpus of Sign Language Video and Text Translation based on JW Bible Verses"""
from __future__ import annotations

import csv
import cv2
import gzip
import json
import math
import numpy as np
from os import path
from copy import copy
from typing import Dict, Any, Set, Optional, List

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.io.gfile import GFile

from pose_format.numpy import NumPyPoseBody
from pose_format.utils.openpose import load_openpose, OpenPoseFrames
from pose_format.pose import Pose

from ..warning import dataset_warning
from ...datasets.config import SignDatasetConfig
from ...utils.features import PoseFeature

try:
    from typing import Literal
except ImportError:
    from typing_extensions import 

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

# This `jws.json` file adapted from the file created using `create_index.py.`
# TODO Make sure to follow up on the adaptation. it should contain "links" to mp4 videos and json text. 
INDEX_URL = "TODO"


_KNOWN_SPLITS = {
    "3.0.0-jw-verse": path.join(path.dirname(path.realpath(__file__)), "splits", "split.3.0.0-jw-verse.json")
}

def load_split(split_name: str): -> Dict[str, List[str]] 
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
        #JWSignConfig(name="verses", include_video=False, include_pose=None, data_type="verse"),

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
            features["fps"] = tf.float32
            features["video"] = self._builder_config.video_feature((1024, 960))
            features["video_path"] = tfds.features.Text()

        
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

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)

        index_path = dl_manager.download(INDEX_URL)
        
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        
        # No need of this infos
        #for datum in index_data.values():
        #    del datum["duration"]
        #    del datum["signer"]
            

        # Don't download videos if not necessary
        if not self._builder_config.include_video:
            for datum in index_data.values():
                del datum["video"]

        # Don't download openpose poses if not necessary
        if self._builder_config.include_pose != "openpose":
            for datum in index_data.values():
                del datum["openpose"]

        # Don't download holistic poses if not necessary
        if self._builder_config.include_pose != "holistic":
            for datum in index_data.values():
                del datum["holistic"]
                
                
                
        
        #urls = {url: url for datum in index_data.values() for url in datum.values() if url is not None}

        #local_paths = dl_manager.download(urls) #TOASK but why even are you downloading everything in the same folder.

        #data = {_id: {k: local_paths[v] if v is not None else None for k, v in datum.items()} for _id, datum in index_data.items()}
        data = index_data
        
        
            
        if self._builder_config.split is not None:
            split = load_split(self._builder_config.split)

            train_args = {"data": data, "split": split["train"]}
            dev_args = {"data": data, "split": split["dev"]}
            test_args = {"data": data, "split": split["test"]}

            return [
                tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs=train_args),
                tfds.core.SplitGenerator(name=tfds.Split.VALIDATION, gen_kwargs=dev_args),
                tfds.core.SplitGenerator(name=tfds.Split.TEST, gen_kwargs=test_args),
            ]

        else:
            return [tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"data": data})]
         

    def _generate_examples(self, , data, split: List[str] | Dict[str, List[str]] = None):
        """ Yields examples. """
        
        default_video = np.zeros((0, 0, 0, 3))  # Empty video
        
        for verse_id, datum in list(data.items()):
            if split is not None and verse_id not in split:
                continue

            features = {
                "id": verse_id,
                "text_path": str(datum["text"]),
            }
            
            # if you want to work with videos   
            if self._builder_config.include_video:
                features["video_path"] = datum["video"]
              
            # if you want to work with poses  
            if self._builder_config.include_pose is not None:
                if self._builder_config.include_pose == "openpose":
                    features["poses_path"] = str(datum["openpose"])

                if self._builder_config.include_pose == "holistic":
                    features["poses_path"] = str(datum["holistic"])                
        
            features["id"] = verse_id
            yield verse_id, features
