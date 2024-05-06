# VSCode Devcontainer Setup Guide

This guide will walk you through the steps to set up and use the [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) configuration in this directory to start a VSCode devcontainer.

Note: This is confirmed working on Apple M series chips. Need to confirm it works on Intel but it should...

## Prerequisites

Before you begin, make sure you have the following installed/configured on your machine:

- [Docker](https://www.docker.com/)
- [AWS Credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html) (Make sure it is in the standard location)

## Getting Started

To get started with the devcontainer, follow these steps:

1. Open Visual Studio Code.

2. If prompted, install the recommended extensions for the devcontainer.

3. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) to open the command palette.

4. Type `Dev Containers: Reopen in Container` and select `dolma-container` from the list.

5. Wait for the devcontainer to build and start. This may take a few minutes the first time you build the image.

6. Once the devcontainer is up and running, you can start working on your project inside the devcontainer.

7. Open a terminal within VSCode for all cli related tasks. (ie: `make test`) Simplest option is `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) and select `Terminal: Create New Terminal`

8. To run a profiler in the devcontainer, you'll need to add  the line ' "runArgs": ["--privileged","--cap-add=SYS_ADMIN"] ' to devcontainer.json before building. 

## Additional Configuration

If you need to customize the devcontainer configuration, you can modify the `.devcontainer/devcontainer.json` file in this directory. Refer to the [VSCode Remote - Containers documentation](https://code.visualstudio.com/docs/remote/containers) for more information on configuring devcontainers. Just be selective about merging changes that might impact correctness across platforms/OS.

Happy hacking!

