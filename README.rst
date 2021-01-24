NFL Slack Bot
=============

*Deploys a continually updated NFL prediction model*

Quick start
-----------

Install the project requirements with pip::

   pip3 install -r requirements.txt

then initialize and populte the sqlite games database::

  python data

Once the game data is synced, called the predict script: ::

  python predict
