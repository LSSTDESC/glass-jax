Template for GLASS extension packages
=====================================

This repository contains a template for how GLASS extension packages can be
structured.

To get started, you can either [use this template] to create a new GitHub
project, or clone the repository to manually set up the new project.

As a rule, a good project name is `glass-MODULE`, where `MODULE` is the name of
the extension module you are creating.  This corresponds to the `import
glass.MODULE` statement you will be using in Python.

You also need to change the name in [setup.cfg](setup.cfg), in multiple places
(search for "extension").  Most importantly, you need to rename the Python
module from [glass/extension.py](glass/extension.py) to your module name.  As
usual, if your module is large and spans multiple files, you can create a
subfolder for it.

Lastly, don't forget to change the author information in [setup.cfg](setup.cfg),
the [LICENSE.txt](LICENSE.txt) file, and the header of the now-renamed
[glass/extension.py](glass/extension.py) module.

[use this template]: https://github.com/glass-dev/glass-extension-template/generate
