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
"""Create an extraction manifest for thinLTO corpora

This script is designed to create a manifest of the *.thinlto.bc and
*.3.import.bc files that need to be moved in order to create an MLGO corpus
from a project built with thinLTO. This script is designed to work with
more complicated build systems, particularly those using archives in the
build process, which can't be handled properly by just looking at the
compilation command database.

Usage:
PYTHONPATH=$PYTHONPATH:. \
  python3 compiler_opt/tools/create_thinlto_extraction_manifest.py \
  --ninja_path=/path/to/ninja \
  --compilation_database=/path/to/compile_commands.json \
  --build_path=/path/to/build \
  --manifest_path=/path/to/output.json

It is not necessary to pass the ninja_path flag. If it is not set, the default
ninja binary will be used.

In order to use this script, the build needs to be configured to use ninja and
ninja needs to be set to keep response files during the build. This can be
achieved using the -d keeprsp flag.
"""

import subprocess
import json
import os

from absl import app
from absl import flags

from compiler_opt.tools import extraction_utils

flags.DEFINE_string('ninja_path', 'ninja',
                    'The path to the ninja executable to use')
flags.DEFINE_string('compilation_database', None,
                    'The path to the compile_commands.json file for the build')
flags.DEFINE_string('build_path', None,
                    'The path to the root of the build being used')
flags.DEFINE_string('manifest_path', 'extraction_manifest.json',
                    'The location of the outputted extraction manifest')

flags.mark_flag_as_required('compilation_database')
flags.mark_flag_as_required('build_path')

FLAGS = flags.FLAGS


def get_archive_list(ninja_path, build_path):
  archive_list = []
  list_targets_command = [ninja_path, '-t', 'targets', 'all']
  with subprocess.Popen(
      list_targets_command, stdout=subprocess.PIPE,
      cwd=build_path) as list_targets_process:
    stdout = list_targets_process.communicate()
    decoded_stdout = stdout[0].decode('UTF-8')
    for target in decoded_stdout.splitlines():
      if 'alink' in target:
        archive_list.append(target.split(':')[0])
  return archive_list


def get_objects_from_rsp_file(archive_path):
  with open(archive_path + '.rsp', encoding='UTF-8') as response_file:
    raw_objects = response_file.read()
    return raw_objects.split(' ')


def load_objects_from_compilation_db(compilation_db_path):
  objects_list = []
  with open(compilation_db_path, encoding='UTF-8') as compilation_db:
    for object_file in extraction_utils.load_from_compile_commands(
        json.load(compilation_db), ''):
      objects_list.append(object_file.relative_output_path())
  return objects_list


def get_objects_in_archives(archive_list, build_dir):
  objects_map = {}
  for archive in archive_list:
    # Getting objects from the rsp file works assuming the build system is
    # using response files instead of passing in the file names on the command
    # line. This assumption is maintained through the Chromium build process,
    # but this logic might need to be modified in the future to extend it to
    # more generic corpora
    objects_in_archive = get_objects_from_rsp_file(
        os.path.join(build_dir, archive))
    for object_file in objects_in_archive:
      objects_map[object_file] = archive
  return objects_map


def main(_):
  extraction_manifest = []

  archive_list = get_archive_list(FLAGS.ninja_path, FLAGS.build_path)
  objects_archive_map = get_objects_in_archives(archive_list, FLAGS.build_path)
  objects_list = load_objects_from_compilation_db(FLAGS.compilation_database)

  for object_file in objects_list:
    if object_file in objects_archive_map:
      base_object_name = os.path.basename(object_file)
      extraction_manifest.append({
          'bitcode':
              objects_archive_map[object_file] +
              f'({base_object_name} at [0-9]+).3.import.bc',
          'index':
              objects_archive_map[object_file] +
              f'({base_object_name} at [0-9]+).thinlto.bc',
          'object': object_file
      })
    else:
      extraction_manifest.append({
          'bitcode': object_file + '.3.import.bc',
          'index': object_file + '.thinlto.bc',
          'object': object_file
      })

  with open(FLAGS.manifest_path, 'w', encoding='UTF-8') as manifest:
    manifest.write(json.dumps(extraction_manifest, indent=4))


if __name__ == '__main__':
  app.run(main)
