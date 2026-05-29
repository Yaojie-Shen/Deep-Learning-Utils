Overview
========

Import style
------------

`dl_utils` exposes many utilities. You can either import from the top-level package:

.. code-block:: python

   from dl_utils import load_mask, save_mask, mask_iou

or import from specific submodules for clarity:

.. code-block:: python

   from dl_utils.mask_utils import load_mask, save_mask, mask_iou


Where to look
-------------

Utilities are grouped roughly by domain:

- Data IO & processing: ``dl_utils.data`` (arrays, normalization, image/video helpers, LMDB, save/load).
- Distributed helpers: ``dl_utils.distributed``.
- Filesystem helpers: ``dl_utils.fs``.
- Inference helpers: ``dl_utils.inference``.
- Misc utilities: top-level modules like ``dl_utils.timer``, ``dl_utils.mask_utils``.


Optional extras
--------------------

Some inference utilities require optional dependencies:

- Ray helpers: install with ``pip install dl-utils[ray]``.
- Ollama helpers: install with ``pip install dl-utils[ollama]``.

If you only need the core utilities, a plain ``pip install dl-utils`` is enough.
