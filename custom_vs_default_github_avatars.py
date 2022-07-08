"""Custom vs. default GitHub avatars dataset"""

from pathlib import Path
from typing import List

import datasets
from datasets.tasks import ImageClassification


logger = datasets.logging.get_logger(__name__)

_URL = "https://huggingface.co/datasets/codecrafters/github-avatars/resolve/main/custom-and-default-avatars.zip"

_HOMEPAGE = "https://codecrafters.io"

_DESCRIPTION = "A dataset of custom vs. default GitHub avatars"


class CatsVsDogs(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "labels": datasets.features.ClassLabel(names=["custom", "default"]),
                }
            ),
            supervised_keys=("image", "labels"),
            task_templates=[
                ImageClassification(image_column="image", label_column="labels")
            ],
            homepage=_HOMEPAGE,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        images_path = (
            Path(dl_manager.download_and_extract(_URL)) / "custom-and-default-avatars"
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"images_path": images_path}
            ),
        ]

    def _generate_examples(self, images_path):
        logger.info("generating examples from = %s", images_path)
        for i, filepath in enumerate(images_path.glob("**/*.png")):
            yield str(i), {
                "image": str(filepath),
                "labels": filepath.parent.name.lower(),
            }
