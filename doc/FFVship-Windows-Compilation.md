## Introduction

Compiling FFVship on windows has been relatively complex due to lib placement not being known to compiler, zimg being non trivial to build and include files missing. As such, I upload below a dev kit containing pre compiled dependencies.

[Windows_devkit.zip]([Windows_devkit](/attachments/cec83fa2-b410-4c25-9882-47e5460d1afa))

## Steps

- Unzip the devkit to get a folder containing various files

- Copy these files and dump them in vship folder

- Use `make buildFFVSHIP` (amd) or `make buildFFVSHIPcuda` (cuda) or `make buildFFVSHIPall` (amd all arch) or `make buildFFVSHIPcudaall` (cuda all arch) depending on your system and goal
