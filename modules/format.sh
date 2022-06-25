#!/bin/bash

ARGS="
--keep-one-line-blocks --keep-one-line-statements --indent=spaces=2
--indent-after-parens --style=linux --max-code-length=80 --break-after-logical
--convert-tabs --suffix=none --recursive
"

astyle $ARGS "./source/*.c"
astyle $ASGS "./include/*.h"
