

```bash
docker run --rm -it -v $PWD:/ws -w /ws --ipc=host python:3.9-slim-bullseye /bin/bash

pip install torch --index-url https://download.pytorch.org/whl/cpu

python -c "import torch; print(torch.__config__.show())"
```

