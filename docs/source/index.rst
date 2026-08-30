mosaic_multigrid
================

.. raw:: html

   <a href="https://pypi.org/project/mosaic-multigrid/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/mosaic-multigrid.svg">
   </a>
   <a href="https://github.com/Abdulhamid97Mousa/mosaic_multigrid">
        <img alt="GitHub" src="https://img.shields.io/github/stars/Abdulhamid97Mousa/mosaic_multigrid?style=social">
   </a>
   <a href="https://github.com/Abdulhamid97Mousa/mosaic_multigrid/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green.svg">
   </a>
   <a href="https://www.python.org/downloads/">
        <img alt="Python" src="https://img.shields.io/badge/python-3.9+-blue.svg">
   </a>

.. raw:: html

   <br><br>

A Gymnasium multi-agent gridworld package with sport-themed environments:
Basketball, American Football, Soccer, and Collect. 81 environments registered
across solo, cooperative, symmetric competitive, and asymmetric competitive
matchups.

.. raw:: html

   <video autoplay loop muted playsinline width="100%"
          style="border-radius: 6px; margin: 1em 0;">
     <source src="_static/videos/demo_mosaic_multigrid.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

.. code-block:: bash

   pip install mosaic_multigrid

.. code-block:: python

   import gymnasium as gym
   import mosaic_multigrid.envs

   env = gym.make("MosaicMultiGrid-BB-2v2-IndAgObs-v1")
   obs, info = env.reset(seed=42)

.. toctree::
   :maxdepth: 2

   documents/installation
   documents/environments
   documents/release_notes
   documents/contributing
   documents/citation
