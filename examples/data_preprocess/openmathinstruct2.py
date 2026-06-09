# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess OpenMathInstruct-2 to verl's RL parquet format.
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


DATA_SOURCE = "nvidia/OpenMathInstruct-2"
REWARD_DATA_SOURCE = "math_dapo"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def make_map_fn(split: str):
    def process_fn(example, idx):
        problem = str(example["problem"]).strip()
        expected_answer = str(example["expected_answer"]).strip()

        return {
            "data_source": REWARD_DATA_SOURCE,
            "prompt": [{"role": "user", "content": f"{problem} {INSTRUCTION}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": expected_answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "dataset": DATA_SOURCE,
                "problem_source": example.get("problem_source"),
                "generated_solution": example.get("generated_solution"),
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="Deprecated. Use --local_save_dir instead.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument("--dataset_name", default=DATA_SOURCE)
    parser.add_argument("--train_split", default="train_1M")
    parser.add_argument("--validation_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local_save_dir",
        default="~/data/openmathinstruct2",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path, split=args.train_split)
    else:
        dataset = datasets.load_dataset(args.dataset_name, split=args.train_split)

    validation_size = min(args.validation_size, max(len(dataset) - 1, 1))
    split_dataset = dataset.train_test_split(test_size=validation_size, seed=args.seed, shuffle=True)
    train_dataset = split_dataset["train"].map(function=make_map_fn("train"), with_indices=True)
    test_dataset = split_dataset["test"].map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(train_dataset[0], f, indent=2)
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(test_dataset[0], f, indent=2)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
