# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract IR for training.

Extract IR for training, either from a compile_commands.json file produced by
cmake, a linker parameter list file, or a thinLTO extraction manifest.

Only run with
'python compiler_opt/tools/extract_ir.py ...'

The compilation is assumed to have been performed with clang, using
-fembed-bitcode=all passed to cc1 (i.e. pass clang -Xclang=-fembed-bitcode=all)

In a distributed ThinLTO case, the compilation is assumed to have been performed
specifying -mllvm -lto-embed-bitcode=post-merge-pre-opt.

In a local ThinLTO case, the compilation is assumedto have been performed
specifying -Wl,--save-temps=import -Wl,--thinlto-emit-index-files
"""

import json
import multiprocessing
import os
import pathlib

from typing import List, Optional

from absl import app
from absl import flags
from absl import logging

from compiler_opt.rl import constant
from compiler_opt.tools import extraction_utils

flags.DEFINE_string(
    'input', None,
    'Input file - either compile_commands.json, linker parameter list, or a'
    'thinLTO extraction manifest.')
flags.DEFINE_enum(
    'input_type', 'json', ['json', 'params', 'manifest'],
    'Input file type - json, params, or manifest. The second refers to lld'
    'params, and the third refers to a thinLTO extraction manifest.')
flags.DEFINE_string('output_dir', None, 'Output directory')
flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel workers for objcopy. `None` for maximum available.')
flags.DEFINE_string('llvm_objcopy_path', 'llvm-objcopy', 'Path to llvm-objcopy')
flags.DEFINE_string(
    'obj_base_dir', '',
    'Base directory for object files. Defaults to current working dir.')
flags.DEFINE_string(
    'cmd_filter', None,
    'Include only those modules with a command line matching this regexp. '
    'Setting it to None for not filtering. Note that the regexp is applied '
    'independently for each separate command line option. For example, ^-Oz$ '
    'will match Oz - built binaries. Does not work with thinlto_build=lld.')
flags.DEFINE_enum(
    'thinlto_build', None, ['distributed', 'local'],
    'Set if the build was performed with either \'distributed\' or '
    '\'local\' ThinLTO. This ensures the thinlto.bc files are also copied. '
    'The build is assumed to have had '
    '-mllvm -lto-embed-bitcode=post-merge-pre-opt passed in the distributed '
    'case, or -Wl,--save-temps=import and -Wl,--thinlto-emit-index-files '
    'passed in the local case.')

FLAGS = flags.FLAGS


def load_from_lld_params(
    params_array: List[str], obj_base_dir: str,
    output_dir: str) -> List[extraction_utils.TrainingIRExtractor]:
  """Create an ObjectFile array based on lld's parameters."""
  # yank out -o and the output. After that, anything not starting with '-', and
  # ending in a '.o', is an object file.
  try:
    minus_o_idx = params_array.index('-o')
    del params_array[minus_o_idx:minus_o_idx + 2]
    just_obj_paths = [
        o for o in params_array if not o.startswith('-') and o.endswith('.o')
    ]
  except ValueError:
    logging.info('This params file does not have an explicit -o option.')
    just_obj_paths = params_array

  def make_obj(obj_file: str) -> extraction_utils.TrainingIRExtractor:
    return extraction_utils.TrainingIRExtractor(
        obj_relative_path=obj_file,
        output_base_dir=output_dir,
        obj_base_dir=obj_base_dir)

  return [make_obj(obj_file) for obj_file in just_obj_paths]

def load_from_manifest(manifest, obj_base_dir, output_dir):
  to_return = []
  for object_file in manifest:
    file_extractor = extraction_utils.TrainingIRExtractor(
      obj_relative_path=object_file['object'],
      output_base_dir=output_dir,
      obj_base_dir=obj_base_dir)
    file_extractor.set_lld_src_bc(object_file['bitcode'])
    file_extractor.set_lld_src_thinlto(object_file['index'])
  return to_return


def load_for_lld_thinlto(
    obj_base_dir: str,
    output_dir: str) -> List[extraction_utils.TrainingIRExtractor]:
  # .3.import.bc is the suffix attached to post-merge-pre-opt ('postimport')
  # IR bitcode saved by lld. It is hardcoded into lld. ThinLTO index files
  # are also emitted next to the postimport bitcode, with the suffix
  # .thinlto.bc instead
  paths = [str(p) for p in pathlib.Path(obj_base_dir).glob('**/*.3.import.bc')]

  def make_spec(obj_file: str):
    return extraction_utils.TrainingIRExtractor(
        # Cut away .3.import.bc
        obj_relative_path=os.path.relpath(obj_file, start=obj_base_dir)[:-12],
        output_base_dir=output_dir,
        obj_base_dir=obj_base_dir)

  return [make_spec(path) for path in paths]


# This is here just for readability, lint complains if the pooling expression is
# over 3 lines; and it needs to be a non-local so it may be pickled.
def extract_artifacts(
    obj: extraction_utils.TrainingIRExtractor) -> Optional[str]:
  return obj.extract(FLAGS.llvm_objcopy_path, FLAGS.cmd_filter,
                     FLAGS.thinlto_build)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  flags.mark_flags_as_required(['output_dir'])

  objs = []
  if FLAGS.input is None:
    if FLAGS.thinlto_build != 'local':
      raise ValueError('--input or --thinlto_build=local must be provided')
    objs = load_for_lld_thinlto(FLAGS.obj_base_dir, FLAGS.output_dir)
  elif FLAGS.input_type == 'json':
    with open(FLAGS.input, encoding='utf-8') as f:
      objs = extraction_utils.load_from_compile_commands(
          json.load(f), FLAGS.output_dir)
  elif FLAGS.input_type == 'params':
    if not FLAGS.obj_base_dir:
      logging.info(
          '-obj_base_dir is unspecified, assuming current directory.'
          'If no objects are found, use this option to specify the root'
          'directory for the object file paths in the input file.')
    with open(FLAGS.input, encoding='utf-8') as f:
      objs = load_from_lld_params([l.strip() for l in f.readlines()],
                                  FLAGS.obj_base_dir, FLAGS.output_dir)
  elif FLAGS.input_type == 'manifest':
    with open(FLAGS.input, encoding='utf-8') as f:
      objs = load_from_manifest(json.load(f))
  else:
    logging.error('Unknown input type: %s', FLAGS.input_type)

  with multiprocessing.Pool(FLAGS.num_workers) as pool:
    relative_output_paths = pool.map(extract_artifacts, objs)

  # This comes first rather than later so global_command_override is at the top
  # of the .json after being written
  if FLAGS.thinlto_build == 'local':
    corpus_description = {
        'global_command_override': constant.UNSPECIFIED_OVERRIDE
    }
  else:
    corpus_description = {}

  corpus_description.update({
      'has_thinlto': FLAGS.thinlto_build is not None,
      'modules': [path for path in relative_output_paths if path is not None]
  })

  with open(
      os.path.join(FLAGS.output_dir, 'corpus_description.json'),
      'w',
      encoding='utf-8') as f:
    json.dump(corpus_description, f, indent=2)

    logging.info('Converted %d files out of %d',
                 len(objs) - relative_output_paths.count(None), len(objs))


if __name__ == '__main__':
  app.run(main)
