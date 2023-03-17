scml-vis
========

|ci| |documentation| |pypi version| |gitter|

A simple visualiser for SCML worlds and tournaments. You can watch `a
demo on YouTube <https://youtu.be/BCDjnnSmIsk>`__ |Alt text| ##
Screenshots |Screen Shot 1| |Screen Shot 2|

Main Features
-------------

-  Displays any world/tournament run using the `SCML
   package <https://www.github.com/yasserfarouk/scml>`__
-  Allows filtering using worlds, agent types, and agent instances
-  Shows world statistics, agent type and instance statistics and
   contract statistics as functions of simulation step/time

Requirements
------------

-  scml-vis requires Python 3.8 or above.
-  scml-vis can visualize worlds created by
   `scml <https://github.com/yasserfarouk/scml>`__ 0.4.2 or later
   (simulations created using older versions of scml can be visualized
   in most cases but are not officially supported).

Installation
------------

With ``pip``:

.. code:: bash

   python3 -m pip install scml-vis

With ``pipx``:

.. code:: bash

   python3 -m pip install --user pipx
   pipx install scml-vis

Usage
-----

The visualizer can be run using any of the following commands:

.. code:: bash

   scmlv
   scmlvis
   scml-vis

Hereafter we will use the shorter version.

-  To visualize any of the recently run worlds and tournaments just run:
   ``bash     scmlv show`` This will open your browser and allow you to
   choose a world or a tournament to display. |show without parameters|

   -  If this is the first time you visualize logs in this folder, you
      will be asked to compile visualization data |compile visualization
      data|
   -  If visualization data is available (or your pressed the
      ``compile visualization data`` button), you can now start
      visualizing the logs |image1|

-  To visualize the logs in a specific folder, you can directly pass the
   folder as in:

   .. code:: bash

      scmlv show -f path-to-your-folder

-  It is also possible to just compile visualization data without
   running the visualizer using:

   .. code:: bash

      scmlv compile path-to-your-folder

-  The visualizer creates a database that it uses to create all the
   figures you see. You can directly explore this database using:

   .. code:: bash

      scmlv explore path-to-your-folder

   This will open a
   `datasette <https://docs.datasette.io/en/stable/getting_started.html>`__
   page allowing you to explore this database |datasette| This dataset
   will contain 8 tables describing everything that was logged in the
   world or tournament. |image2| Please consult `datasette
   documentation <https://docs.datasette.io/en/stable/getting_started.html>`__
   for all the ways you can interact with this dataset.

Available visualizations
------------------------

To visualize your logs, you need to follow three steps:

1. Filter the dataset using the ``Data Selection`` section of the
   sidebar.
2. Choose the **family** of figures you would like to show from the
   ``Figure Selection`` dropdown in the sidebar. Currently we provide
   ``Time-series``, ``Tables`` and ``others`` that are always available
   as well as ``Networks`` that are only available when you visualize a
   single world or filter the tournament to focus on a single world.
3. Choose the specific graph you want to see from the family selected in
   the previous step.

You can watch `a demo of this process
here <https://youtu.be/BCDjnnSmIsk>`__

TODO List (Good Ideas for PRs)
------------------------------

-  [STRIKEOUT:Show negotiation logs (i.e. negotiation results)]
-  [STRIKEOUT:Display all contracts (i.e. in a table) based on selection
   criteria]
-  [STRIKEOUT:Zoom on negotiation details (i.e. exchanged offers)]
-  [STRIKEOUT:Add dynamic figures using plotly/altair]
-  [STRIKEOUT:Add networkx like graphs of contracts / negotiations /
   offers]
-  [STRIKEOUT:Allow starting the app without specifying a folder.]
-  [STRIKEOUT:Add new figure types that do not have time/step in the
   x-axis.]
-  [STRIKEOUT:Correcting the placement of weights on edges in network
   views.]
-  Adding a graph showing negotiation history in the ufun-space of
   negotiators (will require a change in the scml package).
-  Add saving and loading of the visualizer’s state (i.e. what is
   visible).
-  Resolving the strange behavior of CI bands in plotly in some cases.

.. |ci| image:: https://github.com/yasserfarouk/scml-vis/actions/workflows/main.yml/badge.svg
   :target: https://github.com/yasserfarouk/scml-vis/actions/workflows/main.yml
.. |documentation| image:: https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat
   :target: https://scml-vis.github.io/scml-vis/
.. |pypi version| image:: https://img.shields.io/pypi/v/scml-vis.svg
   :target: https://pypi.org/project/scml-vis/
.. |gitter| image:: https://badges.gitter.im/join%20chat.svg
   :target: https://gitter.im/scml-vis/community
.. |Alt text| image:: https://img.youtube.com/vi/BCDjnnSmIsk/0.jpg
   :target: https://youtu.be/BCDjnnSmIsk
.. |Screen Shot 1| image:: docs/shot1.png
.. |Screen Shot 2| image:: docs/shot2.png
.. |show without parameters| image:: docs/show.png
.. |compile visualization data| image:: docs/compile.png
.. |image1| image:: docs/aftercompile.png
.. |datasette| image:: docs/datasette.png
.. |image2| image:: docs/datasettelarge.png

