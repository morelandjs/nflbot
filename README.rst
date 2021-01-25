NFL Slack Bot
=============

*Deploy NFL predictions to a Slack webhook*

Quick start
-----------

Install the project requirements with pip::

   pip3 install -r requirements.txt

then initialize and populte the sqlite games database::

  python3 data.py

Once the game data is synced, called the predict script: ::

  python3 predict.py

See ``python3 predict --help`` for details and options!
