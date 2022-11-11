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
"""Utilities for extracting IR corpora

This module contains common utilities used to extract corpora such as logic
for processing JSON compilation databases.
"""

import os
import re
import shutil
import subprocess

from typing import Dict, List, Optional

from absl import logging


# TODO(ml-compiler-opt): maybe we can also convert here the cmdline file,from a
# \0 - separated list of strings, to a \n one.
def should_include_module(cmdline: str, match_regexp: Optional[str]) -> bool:
  """Determine if the module should be included."""
  if match_regexp is None:
    return True
  lines = cmdline.split('\0')
  return any(len(re.findall(match_regexp, l)) for l in lines)


def get_thinlto_index(cmdline: str, basedir: str) -> Optional[str]:
  opts = cmdline.split('\0')
  for option in opts:
    if option.startswith('-fthinlto-index'):
      return os.path.join(basedir, option.split('=')[1])
  return None


class TrainingIRExtractor:
  """IR and command line extraction from an object file.

  The object file is assumed to have the .llvmbc and .llvmcmd sections.
  """

  def __init__(self, obj_relative_path, output_base_dir, obj_base_dir=None):
    """Set up a TrainingIRExtractor.

    Args:
      obj_relative_path: relative path to the input object file. It will be also
        used to construct the absolute path of the output IR and cmd files, by
        appending it to output_base_dir.
      output_base_dir: the directory under which the output will be produced.
      obj_base_dir: the base directory for all the input object files.
    """
    self._obj_relative_path = obj_relative_path
    self._output_base_dir = output_base_dir
    self._obj_base_dir = obj_base_dir if obj_base_dir is not None else ''
    # .3.import.bc is the suffix attached to post-merge-pre-opt ('postimport')
    # IR bitcode saved by lld. It is hardcoded into lld.
    self._lld_src_bc = os.path.join(self._obj_base_dir,
                                    self._obj_relative_path + '.3.import.bc')
    self._lld_src_thinlto = os.path.join(self._obj_base_dir,
                                         self._obj_relative_path + '.thinlto.bc')

  def obj_base_dir(self):
    return self._obj_base_dir

  def output_base_dir(self):
    return self._output_base_dir

  def relative_output_path(self):
    return self._obj_relative_path

  def input_obj(self):
    return os.path.join(self.obj_base_dir(), self._obj_relative_path)

  def lld_src_bc(self):
    return self._lld_src_bc

  def set_lld_src_bc(self, lld_src_bc):
    self._lld_src_bc = lld_src_bc

  def lld_src_thinlto(self):
    return self._lld_src_thinlto

  def set_lld_src_thinlto(self, lld_src_thinlto):
    self._lld_src_thinlto = lld_src_thinlto

  def dest_dir(self):
    return os.path.join(self.output_base_dir(),
                        os.path.dirname(self._obj_relative_path))

  def module_name(self):
    return os.path.basename(self._obj_relative_path)

  def cmd_file(self):
    return os.path.join(self.dest_dir(), self.module_name() + '.cmd')

  def bc_file(self):
    return os.path.join(self.dest_dir(), self.module_name() + '.bc')

  def thinlto_index_file(self):
    return os.path.join(self.dest_dir(), self.module_name() + '.thinlto.bc')

  def _get_extraction_cmd_command(self, llvm_objcopy_path):
    """Call llvm_objcopy to extract the .llvmcmd section in self._cmd_file."""
    return [
        llvm_objcopy_path, '--dump-section=.llvmcmd=' + self.cmd_file(),
        self.input_obj(), '/dev/null'
    ]

  def _get_extraction_bc_command(self, llvm_objcopy_path):
    """Call llvm_objcopy to extract the .llvmbc section in self._bc_file."""
    return [
        llvm_objcopy_path, '--dump-section=.llvmbc=' + self.bc_file(),
        self.input_obj(), '/dev/null'
    ]

  def _extract_clang_artifacts(self, llvm_objcopy_path: str, cmd_filter: str,
                               is_thinlto: bool) -> Optional[str]:
    """Run llvm-objcopy to extract the .bc and command line."""
    if not os.path.exists(self.input_obj()):
      logging.info('%s does not exist.', self.input_obj())
      return None
    os.makedirs(self.dest_dir(), exist_ok=True)
    try:
      subprocess.run(
          self._get_extraction_cmd_command(llvm_objcopy_path), check=True)
      if cmd_filter is not None or is_thinlto:
        with open(self.cmd_file(), encoding='utf-8') as f:
          lines = f.readlines()
        assert len(lines) == 1
        cmdline = lines[0]
        if not should_include_module(cmdline, cmd_filter):
          logging.info(
              'Excluding module %s because it does not match the filter',
              self.input_obj())
          os.remove(self.cmd_file())
          return None
        if is_thinlto:
          index_file = get_thinlto_index(cmdline, self.obj_base_dir())
          shutil.copy(index_file, self.thinlto_index_file())

      subprocess.run(
          self._get_extraction_bc_command(llvm_objcopy_path), check=True)
    except subprocess.CalledProcessError as e:
      # This may happen if  .o file was build from asm (.S source).
      logging.warning('%s was not processed: %s', self.input_obj(), e)
      return None
    assert (os.path.exists(self.cmd_file()) and
            os.path.exists(self.bc_file()) and
            (not is_thinlto or os.path.exists(self.thinlto_index_file())))
    return self.relative_output_path()

  def _extract_lld_artifacts(self) -> Optional[str]:
    """Extract the .bc file with ThinLTO index from an lld ThinLTO invocation.
    """
    if not os.path.exists(self.lld_src_bc()):
      logging.info('%s does not exist.', self.lld_src_bc())
      return None
    if not os.path.exists(self.lld_src_thinlto()):
      logging.info('%s does not exist.', self.lld_src_thinlto())
      return None
    os.makedirs(self.dest_dir(), exist_ok=True)

    # Copy over the files
    shutil.copy(self.lld_src_bc(), self.bc_file())
    shutil.copy(self.lld_src_thinlto(), self.thinlto_index_file())

    assert os.path.exists(self.bc_file())
    assert os.path.exists(self.thinlto_index_file())
    return self._obj_relative_path

  def extract(self,
              llvm_objcopy_path: Optional[str] = None,
              cmd_filter: Optional[str] = None,
              thinlto_build: Optional[str] = None) -> Optional[str]:
    if thinlto_build == 'local':
      return self._extract_lld_artifacts()
    return self._extract_clang_artifacts(
        llvm_objcopy_path=llvm_objcopy_path,
        cmd_filter=cmd_filter,
        is_thinlto=thinlto_build == 'distributed')


def convert_compile_command_to_objectfile(
    command: Dict[str, str], output_dir: str) -> Optional[TrainingIRExtractor]:
  obj_base_dir = command['directory']
  cmd = command['command']

  cmd_parts = cmd.split()
  try:
    obj_index = cmd_parts.index('-o') + 1
  except ValueError:
    # This could happen if there are non-clang commands in compile_commands.json
    logging.info('Command has no -o option: %s', cmd)
    return None
  obj_rel_path = cmd_parts[obj_index]
  # TODO(mtrofin): is the obj_base_dir correct for thinlto index bc files?
  return TrainingIRExtractor(
      obj_relative_path=obj_rel_path,
      output_base_dir=output_dir,
      obj_base_dir=obj_base_dir)


def load_from_compile_commands(json_array: List[Dict[str, str]],
                               output_dir: str) -> List[TrainingIRExtractor]:
  objs = [
      convert_compile_command_to_objectfile(cmd, output_dir)
      for cmd in json_array
  ]
  # Filter out None, in case there were non-clang commands in the .json
  return [obj for obj in objs if obj is not None]
